import json
import numpy as np
from typing import List
import tempfile
import PIL.Image
from PIL import Image, ImageDraw

import torch

from src.models.unet_3d_condition_edit_cross_attn import UNet3DConditionEditCrossAttnModel, LavieEditCrossAttnModel
from src.models.unet_3d_condition_store_attn import UNet3DConditionAttnStoreModel, LavieAttnStoreModel


model_infos = {
    "zeroscope": {
        "path": "checkpoints/zeroscope_v2_576w",
        "height": 320,
        "width": 576,
        "num_frames": 24
    },
    "lavie": {
        "path": "checkpoints/lavie",
        "height": 320,
        "width": 512,
        "num_frames": 16
    },
}

def extract_conditions_from_script(script_path):
    with open(script_path, 'r') as f:
        script = json.load(f)
    prompt = script["prompt"]
    phrases = script.get("phrases", None)
    bg_phrase = script.get("bg_phrase", None)
    bbox = script.get("bbox", None)
    bbox_proportions = script.get("bbox_proportions", [1])
    camera_speed = script.get("camera_speed", [0.0, 0.0])
    
    return prompt, phrases, bg_phrase, bbox, bbox_proportions, camera_speed

def get_prompt_durations(prompt_proportions, num_frames):
    if prompt_proportions is None:
        return None
    
    prompt_durations = []
    start_frame = 0
    prompt_proportions = np.array(prompt_proportions)
    prompt_frames = np.around(prompt_proportions / np.sum(prompt_proportions) * num_frames).astype(int).tolist()
    for frame in prompt_frames:
        prompt_durations.append([start_frame, start_frame + frame])
        start_frame += frame
    
    return prompt_durations

def interpolate_bbox(bbox, bbox_proportions, num_frames):
    num_objects = len(bbox[0])
    interpolated_data = np.zeros((num_frames, num_objects, 4))
    bbox_proportions = np.array(bbox_proportions)
    box_lengths = np.around(bbox_proportions * num_frames / np.sum(bbox_proportions)).astype(int).tolist()
    cur_start_frame = 0
    for i, box_length in enumerate(box_lengths):
        start_box = bbox[i]
        end_box = bbox[i+1]
        for j in range(box_length):
            w = j / box_length
            interpolated_box = (1 - w) * np.array(start_box) + w * np.array(end_box)
            interpolated_data[cur_start_frame + j] = interpolated_box
        cur_start_frame += box_length

    return interpolated_data.tolist()

def resize_bbox(bbox, height, width):
    new_bbox = []
    for box_frame in bbox:
        new_bbox.append([])
        for box in box_frame:
            x1, y1, x2, y2 = box
            x1 = round(x1 * width)
            y1 = round(y1 * height)
            x2 = round(x2 * width)
            y2 = round(y2 * height)
            new_bbox[-1].append([x1, y1, x2, y2])

    return new_bbox

def load_model_class(model_name, type="edit_cross_attn"):
    if type == "edit_cross_attn":
        if model_name in ["modelscope", "zeroscope"]:
            return UNet3DConditionEditCrossAttnModel
        if model_name in ["lavie"]:
            return LavieEditCrossAttnModel
        
    if type == "store_cross_attn":
        if model_name in ["modelscope", "zeroscope"]:
            return UNet3DConditionAttnStoreModel
        if model_name in ["lavie"]:
            return LavieAttnStoreModel
    
def export_to_gif(image: List[PIL.Image.Image], output_gif_path: str = None, num_frames=-1):
    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif").name

    if num_frames == -1:
        image[0].save(
            output_gif_path,
            save_all=True,
            append_images=image[1:],
            optimize=False,
            duration=125,
            loop=0,
        )
    else:
        image[0].save(
            output_gif_path,
            save_all=True,
            append_images=image[1:num_frames],
            optimize=False,
            duration=125,
            loop=0,
        )

def extract_gif_frames(file_path):
    # Open the gif file
    with Image.open(file_path) as img:
        frames = []
        # Loop over each frame in the gif
        while True:
            # Copy the current frame and append it to the list
            frames.append(img.copy())
            try:
                # Try to move to the next frame
                img.seek(img.tell() + 1)
            except EOFError:
                # Exit the loop if there are no more frames
                break
    return frames

def draw_box_on_video(frames: List[Image.Image], bboxes: List[List[List[int]]]):
    '''
    Draw bounding boxes on each frame of a video.

    Parameters:
        frames (List[Image.Image]): List of PIL Image objects representing video frames.
        bboxes (List[List[List[int]]]): List of bounding box coordinates for each frame.
            Each element in the outer list represents a frame, and each inner list element
            contains bounding boxes for that frame, specified as [x1, y1, x2, y2].

    Returns:
        List[Image.Image]: List of PIL Image objects with bounding boxes drawn.
    '''

    # Loop over each frame and its corresponding set of bounding boxes
    for frame_index, frame_bboxes in enumerate(bboxes):
        draw = ImageDraw.Draw(frames[frame_index])
        
        # Loop over all bounding boxes for the current frame
        for bbox in frame_bboxes:
            xmin, ymin, xmax, ymax = bbox
            draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=1)

    return frames

def tensor_to_pil_images(tensor):
    # Ensure the input tensor is in the expected shape
    assert tensor.ndim == 5, "Tensor should have 5 dimensions (B, 4, f, h, w)"
    assert tensor.size(1) == 4, "Second dimension of tensor should be 4 (representing channels)"
    
    # Rescale tensor from (-1, 1) to (0, 255)
    tensor = (tensor + 1) * 127.5
    tensor = tensor.clamp(0, 255)
    
    # Drop one channel (we'll drop the last channel here)
    tensor = tensor[:, :3, :, :, :]
    
    # Convert tensor to list of PIL images
    images_list = []
    B, C, f, h, w = tensor.shape
    for b in range(B):
        for i in range(f):
            # Get the ith frame for batch b and convert to numpy array
            img_array = tensor[b, :, i, :, :].cpu().byte().numpy()
            # Transpose to (h, w, C) for PIL image creation
            img_array = np.transpose(img_array, (1, 2, 0))
            # Convert to PIL Image
            img = Image.fromarray(img_array, 'RGB')
            images_list.append(img)
    
    return images_list



