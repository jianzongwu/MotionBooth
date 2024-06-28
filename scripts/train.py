import argparse
import logging
from omegaconf import OmegaConf
import os
from datetime import datetime
from einops import rearrange
import gc
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.tensorboard import SummaryWriter
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import export_to_gif

from src.utils import load_model_class, model_infos
from src.pipelines.pipeline_video_model_for_image import VideoModelForImagePipeline
from src.datasets.motionbooth import MotionBoothDataset, VideoDataset, image_video_data_loader
from src.models.unet_3d_condition_store_attn import AttnStore, aggregate_attention
from .utils import save_custom_weights, val_prompt_list, class_name_dict, masked_mse_loss, get_phrase_indices, compute_cross_attn_loss, save_attn_map, save_transformed_images, expand_word_in_text


logger = get_logger(__name__, log_level="INFO")
logging.getLogger('PIL').setLevel(logging.WARNING)


def validation(
    model_path,
    unet,
    config,
    accelerator,
    weight_dtype,
    global_step,
    height,
    width,
    num_frames,
    unique_token,
    class_name,
    log_dir,
):
    logger.info(f"Running validation... \n Generating {len(val_prompt_list)} images and videos")

    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = VideoModelForImagePipeline.from_pretrained(
        model_path,
        unet=accelerator.unwrap_model(unet),
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if config.seed is None else torch.Generator(device=accelerator.device).manual_seed(config.seed)
    save_dir = os.path.join(log_dir, 'val', f'global_step-{global_step}')
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(val_prompt_list)):
        ori_prompt = val_prompt_list[i].format(unique_token, class_name)
        prompt = ori_prompt
        with torch.autocast("cuda"):
            image = pipeline(
                prompt,
                height=height,
                width=width,
                generator=generator,
            ).images[0]
            image.save(os.path.join(save_dir, f"{ori_prompt}.jpg"))

            video = pipeline(
                prompt,
                height=height,
                width=width,
                generator=generator,
                num_frames=num_frames,
            ).videos[0]
            save_path = os.path.join(os.path.join(save_dir, f"{ori_prompt}.gif"))
            export_to_gif(video, save_path)

    del pipeline


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def main(args):
    # Load and merge config
    config = OmegaConf.load(args.config_path)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    args_conf = OmegaConf.create(args_dict)
    config = OmegaConf.merge(config, args_conf)
    
    config_name = args.config_path.split("/")[-1][:-5]
    obj_name = config.obj_name
    if config.exp_name:
        log_folder = f"{config_name}_{obj_name}_{config.exp_name}"
    else:
        log_folder = f"{config_name}_{obj_name}"
    log_dir = os.path.join("logs", log_folder, datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    os.makedirs(log_dir, exist_ok=True)

    OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(config)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set the training seed
    set_seed(config.seed)

    model_name = config.model.model_name
    model_class = load_model_class(model_name, type="store_cross_attn")
    model_path = model_infos[model_name]["path"]
    model_height = model_infos[model_name]["height"]
    model_width = model_infos[model_name]["width"]
    num_frames = model_infos[model_name]["num_frames"]

    class_name = class_name_dict[obj_name]
    unique_token = config.train_data.unique_token
    train_prompt = f"a {unique_token} {class_name}"
    class_prompt = f"a {class_name}"

    # Load tokenizer, text encoder, vae, noise scheduler and models
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    unet = model_class.from_pretrained(model_path, subfolder="unet")

    attn_res_scale_factor=2 ** (len(vae.config.block_out_channels) - 1 + 2)

    # Register store cross attention module
    attn_store = None
    if config.with_cross_attn_loss:
        phrase = f"{unique_token} {class_name}"
        phrase_indices = get_phrase_indices(train_prompt, phrase, tokenizer)
        attn_store = AttnStore()
        unet.register_attn_store_processor(attn_store, attn_res=model_height * model_width // attn_res_scale_factor ** 2, phrase_indices=phrase_indices)

    # Define which components to train
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    trained_parameters = []
    trained_state_dict = {}
    for name, param in unet.named_parameters():
        if config.model.unet_trained_parameters == 'full':
            trained_parameters.append(param)
            trained_state_dict[name] = param
        elif config.model.unet_trained_parameters == 'cross':
            if 'attn2' in name and 'temp_attentions' not in name and 'transformer_in' not in name:
                trained_parameters.append(param)
                trained_state_dict[name] = param
        elif config.model.unet_trained_parameters == 'cross_kv':
            if 'attn2' in name and ('to_k' in name or 'to_v' in name) and 'temp_attentions' not in name and 'transformer_in' not in name:
                trained_parameters.append(param)
                trained_state_dict[name] = param
        elif config.model.unet_trained_parameters == 'wo_temp':
            if 'temp_attentions' not in name and 'temp_convs' not in name:
                trained_parameters.append(param)
                trained_state_dict[name] = param
    logger.info(f"number of trained parameters in unet: {len(trained_parameters)}")

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )
    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if config.scale_lr:
        config.optimizer.learning_rate = (
            config.optimizer.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # Optimizer creation
    optimizer = torch.optim.AdamW(
        trained_parameters,
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        weight_decay=config.optimizer.adam_weight_decay,
        eps=config.optimizer.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = MotionBoothDataset(
        instance_data_root=os.path.join(config.train_data.data_root, obj_name),
        instance_prompt=train_prompt,
        mask_root=os.path.join(config.train_data.mask_root, obj_name) if config.with_mask_loss else None,
        prior_data_root=config.train_data.prior_data_root if config.with_preservation_loss else None,
        prior_data_info_path=config.train_data.prior_data_info_path if config.with_preservation_loss else None,
        tokenizer=tokenizer,
        height=model_height,
        width=model_width,
        num_frames=num_frames,
        padding=config.train_data.padding,
    )

    if config.train_data_video is not None:
        train_video_dataset = VideoDataset(
            data_root=config.train_data_video.data_root,
            data_info_path=config.train_data_video.data_info_path,
            tokenizer=tokenizer,
            height=model_height,
            width=model_width,
            num_frames=num_frames,
        )
    else:
        train_video_dataset = None

    if config.train_data_video is not None:
        train_dataloader = image_video_data_loader(
            train_dataset,
            train_video_dataset,
            config.train_batch_size,
            1,
            accelerator,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
        )
        train_dataloader = accelerator.prepare(train_dataloader)

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        config.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_scheduler.lr_num_cycles,
        power=config.lr_scheduler.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")
        tb_writer = SummaryWriter(log_dir=log_dir)

    # Train!s
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Batch size per device = {config.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # First evaluate on step 0 (before training)
    if config.val_first and accelerator.is_main_process:
        if config.with_cross_attn_loss:
            attn_store.enable(False)
        validation(
            model_path,
            unet,
            config,
            accelerator,
            weight_dtype,
            global_step,
            model_height,
            model_width,
            num_frames,
            unique_token,
            class_name,
            log_dir,
        )
        gc.collect()
        torch.cuda.empty_cache()

    unet.train()
    while global_step < config.max_train_steps:
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                video_length = batch["pixel_values"].shape[2]
                if video_length == 1:
                    if config.with_cross_attn_loss:
                        attn_store.enable(True)
                    if config.with_preservation_loss:
                        pixel_values = torch.cat((batch["pixel_values"], batch["pixel_values_prior"]), dim=0)
                        input_ids = torch.cat((batch["input_ids"], batch["input_ids_prior"]), dim=0)
                        attention_masks = torch.cat((batch["attention_masks"], batch["attention_masks_prior"]), dim=0)
                    else:
                        pixel_values = batch["pixel_values"]
                        input_ids = batch["input_ids"]
                        attention_masks = batch["attention_masks"]
                else:
                    if config.with_cross_attn_loss:
                        attn_store.enable(False)
                    pixel_values = batch["pixel_values"]
                    input_ids = batch["input_ids"]
                    attention_masks = batch["attention_masks"]

                bsz = pixel_values.shape[0]

                # Convert images to latent space
                pixel_values = rearrange(pixel_values, "b c f h w -> (b f) c h w")
                model_input = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                model_input = rearrange(model_input, "(b f) c h w -> b c f h w", f=video_length)
                model_input = model_input * vae.config.scaling_factor

                # Sample noise that we'll add to the model input
                noise = torch.randn_like(model_input)
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = encode_prompt(
                    text_encoder,
                    input_ids,
                    attention_masks,
                )

                # Predict the noise residual
                model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Compute image preservation loss
                preservation_loss = None
                if video_length == 1 and config.with_preservation_loss:
                    model_pred, model_pred_prior = model_pred.chunk(2, dim=0)
                    target, target_prior = target.chunk(2, dim=0)
                    preservation_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # Compute loss
                if config.with_mask_loss and batch.get("obj_masks", None) is not None:
                    loss = masked_mse_loss(model_pred.float(), target.float(), batch["obj_masks"])
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Video loss weight
                if video_length > 1:
                    loss = loss * config.video_loss_weight

                # Compute cross attention loss
                cross_attn_loss = None
                if video_length == 1 and config.with_cross_attn_loss and batch.get("obj_masks", None) is not None:
                    cross_attn = aggregate_attention(attn_store, bsz, model_height // attn_res_scale_factor, model_width // attn_res_scale_factor, 1)
                    if config.with_preservation_loss:
                        cross_attn = cross_attn.chunk(2, dim=0)[0]
                    cross_attn_loss = compute_cross_attn_loss(cross_attn, batch["obj_masks"])
                    loss = loss + cross_attn_loss * config.cross_attn_loss_weight

                    # save the cross attention maps
                    if global_step % config.checkpointing_steps == 0:
                        if config.with_preservation_loss:
                            pixel_values = pixel_values.chunk(2, dim=0)[0]
                        save_dir = os.path.join(log_dir, "cross_attn", f"global_step-{global_step + 1}")
                        os.makedirs(save_dir, exist_ok=True)
                        save_attn_map(cross_attn, os.path.join(save_dir, "attn_map"))
                        save_transformed_images(pixel_values.cpu().detach(), os.path.join(save_dir, "image"))
                        logger.info(f"training images and cross attention maps saved")

                if preservation_loss is not None:
                    loss = loss + preservation_loss * config.preservation_loss_weight

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trained_parameters, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % config.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None and config.checkpoints_total_limit > 0:
                            checkpoints = os.listdir(log_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1][:-4]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(log_dir, removing_checkpoint)
                                    os.remove(removing_checkpoint)

                            save_path = os.path.join(log_dir, f"checkpoint-{global_step}.pth")
                            save_custom_weights(trained_state_dict, save_path)
                            logger.info(f"Saved state to {save_path}")

                    if global_step % config.validation_steps == 0 and not config.no_validation:
                        if config.with_cross_attn_loss:
                            attn_store.enable(False)
                        validation(
                            model_path,
                            unet,
                            config,
                            accelerator,
                            weight_dtype,
                            global_step,
                            model_height,
                            model_width,
                            num_frames,
                            unique_token,
                            class_name,
                            log_dir,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            tb_writer.add_scalar("loss", loss.detach().item(), global_step)
            tb_writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], global_step)
            if cross_attn_loss is not None:
                logs["cross_attn_loss"] = cross_attn_loss.detach().item()
                tb_writer.add_scalar("cross_attn_loss", cross_attn_loss.detach().item(), global_step)
            if preservation_loss is not None:
                logs["preservation_loss"] = preservation_loss.detach().item()
                tb_writer.add_scalar("preservation_loss", preservation_loss.detach().item(), global_step)
            progress_bar.set_postfix(**logs)
            
            if global_step >= config.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info(f"The End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train dreambooth")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--obj_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=1)
    parser.add_argument("--no_validation", action='store_true')
    parser.add_argument("--val_first", action='store_true', help="run validation at 0 step")
    args = parser.parse_args()
    
    main(args)