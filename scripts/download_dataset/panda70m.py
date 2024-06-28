import json
import os
import os.path as osp
import yt_dlp
import asyncio
import pandas as pd
from random import sample
import fire
import shutil
import subprocess

from concurrent.futures import ThreadPoolExecutor


def extract_frames_ffmpeg(video_path, output_dir, timestamps):
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the FFmpeg command
    output_file_pattern = os.path.join(output_dir, '%05d.jpg')
    start_time, end_time = timestamps

    # Construct the FFmpeg command to extract frames
    command = [
        'ffmpeg',
        '-i', video_path,           # Input file
        '-ss', start_time,          # Start time
        '-to', end_time,            # End time
        # '-vf', f'fps=10',           # Extract at source frame rate (or specify fps)
        '-q:v', '0',                # Output quality (scale of 1-31, lower is better)
        output_file_pattern         # Output file pattern
    ]
    
    # Run the command
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ytb_download(uid, url, filtered_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    yt_opts = {
        "format": f"wv*[height>=360][ext=mp4]",
        "outtmpl": osp.join(output_dir, f"{uid}.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }],
    }

    video_path = osp.join(output_dir, f"{uid}.mp4")
    if osp.exists(video_path):
        print(f"{uid} already downloaded.")
        return 0

    try:
        with yt_dlp.YoutubeDL(yt_opts) as ydl:
            ydl.download([url])
        for idx, times in enumerate(filtered_data['timestamps']):
            clip_dir = f"{output_dir}_{idx}"
            extract_frames_ffmpeg(video_path, clip_dir, times)
        shutil.rmtree(output_dir)
        return 0
    except Exception as e:
        print(f"Failed to process {uid}: {str(e)}")
        return -1


async def main(csv_path, max_videos):
    df = pd.read_csv(csv_path)
    df = df.sample(n=max_videos, random_state=0)

    output_root = f"data/panda/random_{max_videos}"
    video_info = {}
    tasks = []

    executor = ThreadPoolExecutor(max_workers=10)
    loop = asyncio.get_event_loop()

    for index, row in df.iterrows():
        timestamps = eval(row["timestamp"])
        captions = eval(row["caption"])
        filtered_data = {"timestamps": [], "captions": []}

        for time, caption in zip(timestamps, captions):
            filtered_data["timestamps"].append(time)
            filtered_data["captions"].append(caption)

        if filtered_data["timestamps"]:
            for idx, _ in enumerate(filtered_data["timestamps"]):
                video_info[f"{row['videoID']}_{idx}"] = filtered_data["captions"][idx]
            output_dir = os.path.join(output_root, row['videoID'])
            task = loop.run_in_executor(executor, ytb_download, row["videoID"], row["url"], filtered_data, output_dir)
            tasks.append(task)

        if len(video_info) >= max_videos:
            break

    # Save the video-captions data to a file
    with open(osp.join("data/panda", f"captions_random.json"), "w") as fp:
        json.dump(video_info, fp, indent=2)
    
    await asyncio.gather(*tasks)

    print("all download completed")


def entry(csv="data/panda/panda70m_training_2m.csv", max_videos=500):
    asyncio.run(main(csv, max_videos))


if __name__ == '__main__':
    fire.Fire(entry)