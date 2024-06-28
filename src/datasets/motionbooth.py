import os
import random
import json
from pathlib import Path
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


class MotionBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        mask_root,
        tokenizer,
        prior_data_root=None,
        prior_data_info_path=None,
        height=320,
        width=576,
        num_frames=24,
        padding=False,
    ):
        self.size = (height, width)
        self.num_frames = num_frames
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = sorted(list(Path(instance_data_root).iterdir()))
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        self.mask_images_path = None
        if mask_root is not None:
            self.mask_images_path = sorted(list(Path(mask_root).iterdir()))
            if len(self.instance_images_path) != len(self.mask_images_path):
                raise ValueError(f"Instance image number does not equal to mask image number")

        if prior_data_root is not None:
            with open(prior_data_info_path, 'r') as f:
                self.prior_data_info = json.load(f)
            self.prior_data_paths = sorted(list(Path(prior_data_root).iterdir()))
            self.num_prior_images = len(self.prior_data_paths)
            self._length = max(self.num_prior_images, self.num_instance_images)
        else:
            self.prior_data_paths = None

        if padding:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize((height, height), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.Pad(((width - height) // 2, 0, (width - height) // 2, 0)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.mask_transforms = transforms.Compose(
                [
                    transforms.Resize((height, height), interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.Pad(((width - height) // 2, 0, (width - height) // 2, 0)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.mask_transforms = transforms.Compose(
                [
                    transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor(),
                ]
            )

        self.prior_image_trasforms = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __len__(self):
        return self._length
    
    def load_random_frame(self, video_folder):
        frame_paths = sorted(list(Path(video_folder).iterdir()))
        selected_frame = random.randint(0, len(frame_paths) - 1)
        frame_path = frame_paths[selected_frame]
        frame = Image.open(frame_path).convert("RGB")
        frame = self.prior_image_trasforms(frame)

        return frame

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["pixel_values"] = self.image_transforms(instance_image).unsqueeze(1)

        if self.mask_images_path:
            mask_image = Image.open(self.mask_images_path[index % self.num_instance_images]).convert("L")
            mask_image = self.mask_transforms(mask_image)
            binary_mask = (mask_image > 0.5).float()
            example["obj_masks"] = binary_mask.unsqueeze(0)

        text_inputs = tokenize_prompt(
            self.tokenizer, self.instance_prompt
        )
        example["input_ids"] = text_inputs.input_ids
        example["attention_masks"] = text_inputs.attention_mask

        if self.prior_data_paths is not None:
            prior_data_path = self.prior_data_paths[index % self.num_prior_images]
            prior_image = self.load_random_frame(prior_data_path)
            example["pixel_values_prior"] = prior_image.unsqueeze(1)

            videoid = prior_data_path.name.split('/')[-1]
            video_prompt = self.prior_data_info[videoid]
            text_inputs = tokenize_prompt(
                self.tokenizer, video_prompt
            )
            example["input_ids_prior"] = text_inputs.input_ids
            example["attention_masks_prior"] = text_inputs.attention_mask

        return example


class MotionBoothVideoDataset(MotionBoothDataset):
    def load_video(self, video_folder):
        frame_paths = sorted(list(Path(video_folder).iterdir()))
        start_frame = random.randint(0, max(0, len(frame_paths) - self.num_frames))
        frames = []
        for frame_path in frame_paths[start_frame:start_frame + self.num_frames]:
            frame = Image.open(frame_path).convert("RGB")
            frame = self.image_transforms(frame)
            frames.append(frame)

        return torch.stack(frames, dim=1), start_frame

    def load_mask_video(self, mask_video_path, start_frame):
        frame_paths = sorted(list(Path(mask_video_path).iterdir()))
        frames = []
        for frame_path in frame_paths[start_frame:start_frame + self.num_frames]:
            frame = Image.open(frame_path).convert("L")
            frame = self.mask_transforms(frame)
            binary_frame = (frame > 0.5).float()
            frames.append(binary_frame)
            
        return torch.stack(frames, dim=1)

    def __getitem__(self, index):
        example = {}

        instance_video_folder = self.instance_images_path[index % self.num_instance_images]
        example["pixel_values"], start_frame = self.load_video(instance_video_folder)

        if self.mask_images_path:
            mask_video_folder = self.mask_images_path[index % self.num_instance_images]
            example["obj_masks"] = self.load_mask_video(mask_video_folder, start_frame)

        text_inputs = tokenize_prompt(
            self.tokenizer, self.instance_prompt
        )
        example["input_ids"] = text_inputs.input_ids
        example["attention_masks"] = text_inputs.attention_mask

        if self.prior_data_root is not None:
            prior_video_folder = self.prior_videos_path[index % self.num_prior_videos]
            example["pixel_values_prior"], _ = self.load_video(prior_video_folder)

            videoid = int(prior_video_folder.name.split('/')[-1])
            video_data = self.prior_data_info[self.prior_data_info['videoid'] == videoid]
            prior_video_prompt = video_data.iloc[0]['name']

            text_inputs = tokenize_prompt(
                self.tokenizer, prior_video_prompt
            )
            example["input_ids_prior"] = text_inputs.input_ids
            example["attention_masks_prior"] = text_inputs.attention_mask

        return example


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        data_info_path=None,
        height=320,
        width=576,
        num_frames=24,
    ):
        self.size = (height, width)
        self.num_frames = num_frames
        self.tokenizer = tokenizer

        self.data_root = data_root
        if data_info_path.endswith("csv"):
            self.data_info_type = "csv"
            self.data_info = pd.read_csv(data_info_path)
        elif data_info_path.endswith("json"):
            self.data_info_type = "json"
            with open(data_info_path, 'r') as f:
                self.data_info = json.load(f)
        
        self.video_names = sorted(list(self.data_info.keys()))
        self.num_videos = len(self.video_names)
        self._length = self.num_videos

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def __len__(self):
        return self._length

    def load_video(self, video_folder):
        frame_paths = sorted(list(Path(video_folder).iterdir()))
        start_frame = random.randint(0, max(0, len(frame_paths) - self.num_frames))
        frames = []
        for frame_path in frame_paths[start_frame:start_frame + self.num_frames]:
            frame = Image.open(frame_path).convert("RGB")
            frame = self.image_transforms(frame)
            frames.append(frame)

        return torch.stack(frames, dim=1)

    def __getitem__(self, index):
        example = {}

        video_name = self.video_names[index % self.num_videos]
        video_folder = os.path.join(self.data_root, video_name)
        example["pixel_values"] = self.load_video(video_folder)
        
        if self.data_info_type == "csv":
            video_data = self.data_info[self.data_info['videoid'] == video_name]
            video_prompt = video_data.iloc[0]['name']
        elif self.data_info_type == "json":
            video_prompt = self.data_info[video_name]
        
        text_inputs = tokenize_prompt(
            self.tokenizer, video_prompt
        )
        example["input_ids"] = text_inputs.input_ids
        example["attention_masks"] = text_inputs.attention_mask

        return example


def image_video_data_loader(dataset_image, dataset_video, batch_size_image, batch_size_video, accelerator):
    loader_image = DataLoader(dataset_image, batch_size=batch_size_image, shuffle=True)
    loader_video = DataLoader(dataset_video, batch_size=batch_size_video, shuffle=True)
    loader_image, loader_video = accelerator.prepare(loader_image, loader_video)
    
    iterator_image = iter(loader_image)
    iterator_video = iter(loader_video)
    
    while True:
        try:
            # Fetch batch from dataset 1
            batch1 = next(iterator_image)
            yield batch1
            # Fetch batch from dataset 2
            batch2 = next(iterator_video)
            yield batch2
        except StopIteration:
            # Restart the iterator if one of the datasets runs out of data
            iterator_image = iter(loader_image)
            iterator_video = iter(loader_video)