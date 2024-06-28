import argparse
import json
import os
import copy
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def visualize_bbox(prompt, phrases, height, width, bbox, camera_speed):
    dpi = 100  # Default DPI, can be adjusted
    num_frames = len(bbox)
    draw_bbox = copy.deepcopy(bbox)
    # update bbox according to camera_speed
    for f in range(num_frames):
        for obj_idx in range(len(draw_bbox[f])):
            draw_bbox[f][obj_idx][0] -= camera_speed[0] * f / num_frames
            draw_bbox[f][obj_idx][1] -= camera_speed[1] * f / num_frames
            draw_bbox[f][obj_idx][2] -= camera_speed[0] * f / num_frames
            draw_bbox[f][obj_idx][3] -= camera_speed[1] * f / num_frames


    fig_width, fig_height = width / dpi, height / dpi  # Convert pixels to inches
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_axis_off()  # Turn off the axes
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    font_size = 24
    font = ImageFont.truetype("fonts/Calibri Regular.ttf", size=font_size)
    text_color = "white"
    background_color = "black"
    
    def update(frame_idx):
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)

        # Draw video title (prompt)
        title_text = prompt
        title_position = (2, 2)
        _, _, text_width, text_height  = draw.textbbox((0, 0), title_text, font=font)
        draw.rectangle([title_position, (title_position[0] + text_width, title_position[1] + text_height)], fill=background_color)
        draw.text(title_position, title_text, fill=text_color, font=font)

        box_frame = draw_bbox[frame_idx]
        for i, box_char in enumerate(box_frame):
            draw.rectangle(box_char, outline="red", width=2)

            # Draw name of the object in the bounding box
            box_name = phrases[i]
            box_top_left = (box_char[0] + 2, box_char[1] + 2)
            _, _, text_width, text_height  = draw.textbbox((0, 0), box_name, font=font)
            draw.rectangle([box_top_left, (box_top_left[0] + text_width, box_top_left[1] + text_height)], fill=background_color)
            draw.text(box_top_left, box_name, fill=text_color, font=font)

        ax.imshow(image)  # Display the image with the bounding box

    ani = animation.FuncAnimation(fig, update, frames=len(draw_bbox), repeat=True)
    return ani


def main(args):
    with open(args.script_path, 'r') as file:
        script = json.load(file)

    prompt = script["prompt"]
    phrases = script["phrases"]
    height = script["height"]
    width = script["width"]
    bbox = script["bbox"]
    camera_speed = script.get("camera_speed", [0.0,0.0])

    ani = visualize_bbox(prompt, phrases, height, width, bbox, camera_speed)

    save_folder = f"results_vis/{prompt}"
    os.makedirs(save_folder, exist_ok=True)
    ani.save(os.path.join(save_folder, "bbox.gif"), writer='pillow', fps=10)  # Save the animation as a GIF

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script_path', type=str, default='data/scripts/barrel_drift_down_river.json')
    parser.add_argument('--model_name', type=str, default='ours')
    args = parser.parse_args()

    main(args)