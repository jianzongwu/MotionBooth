import time
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

import numpy as np
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.utils import BaseOutput
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.text_to_video_synthesis import TextToVideoSDPipeline

from accelerate.logging import get_logger

from src.models.unet_3d_condition_edit_cross_attn import UNet3DConditionEditCrossAttnModel, AttnEditProcessor


logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',)
logger = get_logger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True
printed = [False]


def tensor2vid(video: torch.Tensor, processor, output_type="pil"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs


@dataclass
class TextToVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for text-to-video pipeline.

    Args:
        frames (`List[List[PIL.Image.Image]]`):
            List of denoised PIL images of length `batch_size`.
    """

    frames: List[List[Image.Image]]


class MotionBoothPipeline(TextToVideoSDPipeline):
    r"""
    Pipeline for layout-to-video using MultiDiffusion (Gen-L-Video) with BoxDiff.

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionEditCrossAttnModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def shift_latent(self, latents, camera_speed, bbox, num_shift_steps):
        '''
        mode ('loop', 'sample'):
            'loop': fill in the hole using the opposite pixels, produing loop background.
            'sample':  fill in the hole using pixels randomly sampled from background.
        '''
        batch_size, channels, num_frames, height, width = latents.shape
        shift_speed = [s / self.vae_scale_factor / num_frames for s in camera_speed]
        sx = shift_speed[0] / num_shift_steps
        sy = shift_speed[1] / num_shift_steps
        for f in range(num_frames):
            sfx = sx * f
            sfy = sy * f
            shift_x = int(self.shift_x[f] + sfx) - int(self.shift_x[f])
            shift_y = int(self.shift_y[f] + sfy) - int(self.shift_y[f])
            
            obj_mask = torch.ones_like(latents[0,0,f,:,:], dtype=torch.bool)
            # exclude object latents when sampling background
            if bbox:
                for obj_idx in range(len(bbox[f])):
                    box = bbox[f][obj_idx]
                    box = [int(b / self.vae_scale_factor) for b in box]
                    obj_mask[box[1]:box[3],box[0]:box[2]] = False
            if shift_x != 0:
                fill_x = torch.zeros_like(latents[:,:,f,:,:abs(shift_x)])
                for i in range(height):
                    included_indices = np.array([x for x in range(0, width) if obj_mask[i,x]])
                    sampled_indices = np.random.choice(included_indices, size=abs(shift_x), replace=False)
                    fill_x[:,:,i,:] = latents[:,:,f,i,sampled_indices]
            if shift_y != 0:
                fill_y = torch.zeros_like(latents[:,:,f,:abs(shift_y),:])
                for j in range(width):
                    included_indices = np.array([y for y in range(0, height) if obj_mask[y,j]])
                    sampled_indices = np.random.choice(included_indices, size=abs(shift_y), replace=False)
                    fill_y[:,:,:,j] = latents[:,:,f,sampled_indices,j]

            if shift_x > 0:
                temp = latents[:,:,f,:,shift_x:]
                latents[:,:,f,:,:] = torch.cat([temp, fill_x], dim=-1)
            elif shift_x < 0:
                temp = latents[:,:,f,:,:shift_x]
                latents[:,:,f,:,:] = torch.cat([fill_x, temp], dim=-1)
            if shift_y > 0:
                temp = latents[:,:,f,shift_y:,:]
                latents[:,:,f,:,:] = torch.cat([temp, fill_y], dim=-2)
            elif shift_y < 0:
                temp = latents[:,:,f,:shift_y,:]
                latents[:,:,f,:,:] = torch.cat([fill_y, temp], dim=-2)

            self.shift_x[f] += sfx
            self.shift_y[f] += sfy

            if bbox:
                bbox[f][obj_idx][0] -= shift_x * self.vae_scale_factor
                bbox[f][obj_idx][1] -= shift_y * self.vae_scale_factor
                bbox[f][obj_idx][2] -= shift_x * self.vae_scale_factor
                bbox[f][obj_idx][3] -= shift_y * self.vae_scale_factor

        logger.info(f"all shift_x: {self.shift_x}, all shift_y: {self.shift_y}")
            
        return latents

    def decode_latents(self, latents, decoder_chunk_size=24):
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        frames = []
        for i in range(0, latents.shape[0], decoder_chunk_size):
            frame = self.vae.decode(latents[i : i + decoder_chunk_size]).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        video = (
            frames[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + frames.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video

    def get_token_map(self, prompt, padding="do_not_pad"):
        """Get a list of mapping: prompt index to str (prompt in a list of token str)"""
        fg_prompt_tokens = self.tokenizer([prompt], padding=padding, max_length=77, return_tensors="np")
        input_ids = fg_prompt_tokens["input_ids"][0]

        token_map = []
        for ind, item in enumerate(input_ids.tolist()):
            token = self.tokenizer._convert_id_to_token(item)
            token_map.append(token)

        return token_map

    def get_phrase_indices(
        self,
        prompt,
        phrases,
    ):
        """
        Returns:
            phrases_indices: List[List[int]]
            prompt: str
        """ 
        token_map = self.get_token_map(prompt=prompt, padding="do_not_pad")
        token_map_str = " ".join(token_map)
        logger.info(f"Full str: {token_map_str}")

        phrase_indices = []
        if phrases is None:
            return [None]

        for obj in phrases:
            if obj not in prompt:
                phrase_indices.append(None)
            else:
                phrase_token_map = self.get_token_map(prompt=obj, padding="do_not_pad")
                # Remove <bos> and <eos> in substr
                phrase_token_map = phrase_token_map[1:-1]
                phrase_token_map_len = len(phrase_token_map)
                phrase_token_map_str = " ".join(phrase_token_map)

                logger.info(f"Object: {obj}, object str: {phrase_token_map_str}")

                # Count the number of token before substr
                # The substring comes with a trailing space that needs to be removed by minus one in the index.
                obj_first_index = len(token_map_str[: token_map_str.index(phrase_token_map_str) - 1].split(" "))

                obj_position = list(range(obj_first_index, obj_first_index + phrase_token_map_len))
                phrase_indices.append(obj_position)

        logger.info(f"phrase_indices: {phrase_indices}")

        return phrase_indices

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        phrases: List[str] = None,
        bg_phrase: str = None,
        height: Optional[int] = 320,
        width: Optional[int] = 576,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        num_frames: int = 16,
        decoder_chunk_size: Optional[int] = 24,
        # motion control parameters
        attn_edit_processor: AttnEditProcessor = None,
        bbox: List[int] = None,
        edit_scale: float = 7.5,
        max_amp_steps: int = 5,
        # camera control parameters
        start_shift_step: int = 5,
        max_shift_steps: int = 10,
        camera_speed: List[int] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        decoder_chunk_size = decoder_chunk_size if decoder_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        batch_size = 1
        num_images_per_prompt = 1
        # extract phrase indices according to phrases
        # if phrases not in prompt, add them to the end of prompt
        phrase_indices = self.get_phrase_indices(prompt, phrases)
        bg_phrase_indices = self.get_phrase_indices(prompt, [bg_phrase])[0] if bg_phrase else None

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        input_ids = self.tokenizer(prompt)['input_ids']
        prompt_length = len(input_ids)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self.shift_x = [0] * num_frames
        self.shift_y = [0] * num_frames

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        i = 0
        time_start = time.time()
        while i < len(timesteps):   
            t = timesteps[i] 
            logger.info(f"denoising step {i}")
            if i >= start_shift_step and i < start_shift_step + max_shift_steps and camera_speed:
                latents = self.shift_latent(latents, camera_speed, bbox, max_shift_steps)

            if bbox:
                attn_edit_processor.set_variables(
                    bbox=bbox,
                    phrase_indices=phrase_indices,
                    bg_phrase_indices=bg_phrase_indices,
                    edit_scale=edit_scale,
                    prompt_length=prompt_length,
                    is_amp = i < max_amp_steps,
                )
                attn_edit_processor.enable(True)

            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            noise_pred_text = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample
            attn_edit_processor.enable(False)

            # predict the noise residual
            noise_pred_uncond = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=negative_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            b, c, f, h, w = latents.shape
            latents = latents.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
            noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
            latents_denoised = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            latents = latents_denoised[None, :].reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)
            
            i += 1

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        time_end = time.time()
        execution_time = (time_end - time_start) / 60
        print(f"denoising time: {execution_time:.1f} min")

        # 8. Post-processing
        video_tensor = self.decode_latents(latents, decoder_chunk_size=decoder_chunk_size)
        video = tensor2vid(video_tensor, processor=self.image_processor)
        self.maybe_free_model_hooks()

        return TextToVideoDiffusionPipelineOutput(frames=video)