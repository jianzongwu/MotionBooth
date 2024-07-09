import os
import argparse

import torch

from diffusers.utils import export_to_gif

from accelerate import PartialState

from src.pipelines.pipeline_motionbooth import MotionBoothPipeline
from src.models.unet_3d_condition_edit_cross_attn import AttnEditProcessor
from src.utils import extract_conditions_from_script, model_infos, interpolate_bbox, resize_bbox, load_model_class
from scripts.utils import load_custom_weights
from .visualization.visualize_box_video import visualize_bbox

import warnings
warnings.simplefilter('ignore', UserWarning)


def main(args):
    # Prepare script variables
    prompt, phrases, bg_phrase, bbox, bbox_proportions, camera_speed = extract_conditions_from_script(args.script_path)
    
    num_frames = model_infos[args.model_name]["num_frames"]
    height = model_infos[args.model_name]["height"]
    width = model_infos[args.model_name]["width"]
    
    if bbox:
        bbox = interpolate_bbox(bbox, bbox_proportions, num_frames)
        bbox = resize_bbox(bbox, height, width)
    
    camera_speed[0] = camera_speed[0] * width
    camera_speed[1] = camera_speed[1] * height
    
    distributed_state = PartialState()

    # initialize models, attention store modules, and pipelines
    attn_edit_processor = AttnEditProcessor(
        num_frames=num_frames,
        image_width=width,
        image_height=height,
    )
    
    model_path = model_infos[args.model_name]["path"]
    model_class = load_model_class(args.model_name)
    dtype = torch.float16
    unet = model_class.from_pretrained(
        model_path, subfolder="unet", use_safetensors=False, torch_dtype=dtype
    )
    if args.customize_ckpt_path is not None:
        cus_ckpt_step = args.customize_ckpt_path.split('/')[-1].split('-')[-1][:-4]
        load_custom_weights(unet, args.customize_ckpt_path)
        prompt = prompt.replace("[rare token] [class name]", f"sks {args.class_name}")
        phrases[0] = phrases[0].replace("[rare token]", "sks")
        phrases[0] = phrases[0].replace("[class name]", args.class_name)

    pipe = MotionBoothPipeline.from_pretrained(
        model_path, unet=unet, torch_dtype=dtype
    )
    pipe.to(distributed_state.device)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.unet.register_attn_edit_processor(attn_edit_processor=attn_edit_processor)

    # print script information
    if distributed_state.is_main_process:
        print(f"prompt: {prompt}\nnum_frames {num_frames}\nbbox: {bbox}\ncamera speed: {camera_speed}")

    # create save folder
    script_str = args.script_path.split('/')[-1][:-5]
    if args.class_name is not None:
        script_str += f"_{'_'.join(args.class_name.split(' '))}"
    save_folder = f"results/{script_str}"
    os.makedirs(save_folder, exist_ok=True)

    # draw the bounding box anime
    bbox_anime_path = os.path.join(save_folder, "script_anime.gif")
    if distributed_state.is_main_process and bbox:
        ani = visualize_bbox(prompt, phrases, height, width, bbox, camera_speed)
        ani.save(bbox_anime_path, writer='pillow', fps=10)
        del ani

    with distributed_state.split_between_processes(list(range(args.num_samples))) as sample_indices:
        for idx in sample_indices:
            seed = args.base_seed + idx
            generator = torch.Generator(distributed_state.device).manual_seed(seed)

            result_name = ''
            result_name += f'gs{args.text_guidance_scale}-'
            if camera_speed[0] != 0 or camera_speed[1] != 0:
                result_name += f'cs{int(camera_speed[0])}-{int(camera_speed[1])}-'
                result_name += f'ss{args.start_shift_step}-{args.start_shift_step + args.max_shift_steps}-'
            if bbox is not None:
                result_name += f'es{args.edit_scale}-'
                result_name += f'as{args.max_amp_steps}-'
            if args.customize_ckpt_path is not None:
                result_name += f'cus-{cus_ckpt_step}-'
            result_name += f's{seed}'
            
            save_path = os.path.join(save_folder, result_name + '.gif')

            frames = pipe(
                prompt=prompt,
                phrases=phrases,
                bg_phrase=bg_phrase,
                height=height,
                width=width,
                num_frames=num_frames,
                generator=generator,
                guidance_scale=args.text_guidance_scale,
                # subject control parameters
                bbox=bbox.deepcopy(),
                attn_edit_processor=attn_edit_processor,
                edit_scale=args.edit_scale,
                max_amp_steps=args.max_amp_steps,
                # camera control parameters
                camera_speed=camera_speed,
                start_shift_step=args.start_shift_step,
                max_shift_steps=args.max_shift_steps,
            ).frames[0]

            export_to_gif(frames, save_path)
            print(f"video '{save_path}' saved")

    if distributed_state.is_main_process:
        print("The End")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script_path', help='path of script json file', required=True)
    parser.add_argument('--model_name', help='name of T2V base model [zeroscope, lavie]', required=True)
    # customize parameters
    parser.add_argument('--customize_ckpt_path', type=str, help='trained customize model path', default=None)
    parser.add_argument('--class_name', help='class name of the customized subject', default=None)
    # T2V parameters
    parser.add_argument('--num_samples', type=int, help='num of generated samples', default=8)
    parser.add_argument('--text_guidance_scale', type=float, default=7.5)
    parser.add_argument('--base_seed', type=int, help='base random seed', default=0)
    # motion control parameters
    parser.add_argument('--edit_scale', type=float, help="edit scale of cross attn map", default=10.0)
    parser.add_argument('--max_amp_steps', type=int, help="max steps to perform attention amplification", default=15)
    # camera control parameters
    parser.add_argument('--start_shift_step', type=int, help='step to start latent shift', default=10)
    parser.add_argument('--max_shift_steps', type=int, help='max latent shift steps', default=10)
    args = parser.parse_args()

    main(args)