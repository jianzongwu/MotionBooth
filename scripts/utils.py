from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

val_prompt_list = [
'a {0} {1}',
'a {0} {1} walking in the street',
'a {0} {1} skiing on snow',
'a {0} {1} surfing on the wave',
'a {0} {1} running on the grass',
'a {0} {1} running on the beach',
'a {0} {1} running in the forest',
'a {0} {1} riding a bike on the road',
] 

class_name_dict = {
    "cat2": "cat",
    "dog": "dog",
    "dog2": "dog",
    "dog3": "dog",
    "dog5": "dog",
    "dog6": "dog",
    "dog7": "dog",
    "dog8": "dog",
    "duck_toy": "toy duck",
    "monster_toy": "toy monster",
    "pet_cat1": "cat",
    "pet_cat5": "cat",
    "pet_cat7": "cat",
    "pet_dog1": "dog",
    "plushie_happysad": "plushie happysad",
    "plushie_panda": "plushie panda",
    "plushie_teddybear": "plushie teddybear",
    "plushie_tortoise": "plushie tortoise",
    "rc_car": "toy car",
    "red_cartoon": "cartoon demon",
    "robot_toy": "toy robot",
    "toy_bear": "toy bear",
    "transport_car6": "car",
    "transport_car7": "car",
    "transport_motorbike1": "motorbike",
    "wolf_plushie": "plushie wolf",
}

def masked_mse_loss(model_pred, target, mask):
    # Convert mask to a boolean or use it as float if it's already 0s and 1s
    _, _, _, height, width = model_pred.shape
    mask = mask.squeeze(2)
    mask = F.interpolate(mask, size=(height, width), mode='bilinear', align_corners=False)
    mask = mask.unsqueeze(2)
    mask = mask.bool()

    # Apply the mask by setting non-masked elements to zero
    masked_model_pred = model_pred * mask
    masked_target = target * mask

    # Compute the MSE loss on the masked tensors
    # Note: Since mse_loss computes the mean over all elements and we want the mean only over masked elements,
    # we need to adjust the reduction manually.
    total_loss = F.mse_loss(masked_model_pred, masked_target, reduction='sum')
    
    # Count the number of masked elements
    num_masked_elements = mask.sum()

    # Calculate the average only over masked elements
    if num_masked_elements > 0:
        mse_loss = total_loss / num_masked_elements
    else:
        mse_loss = torch.tensor(0.0).to(model_pred.device)  # If no elements are masked, return zero

    return mse_loss


def get_phrase_indices(prompt, phrase, tokenizer):
    def get_token_map(prompt, padding="do_not_pad"):
        """Get a list of mapping: prompt index to str (prompt in a list of token str)"""
        fg_prompt_tokens = tokenizer([prompt], padding=padding, max_length=77, return_tensors="np")
        input_ids = fg_prompt_tokens["input_ids"][0]

        token_map = []
        for ind, item in enumerate(input_ids.tolist()):
            token = tokenizer._convert_id_to_token(item)
            token_map.append(token)

        return token_map
    
    token_map = get_token_map(prompt=prompt, padding="do_not_pad")
    token_map_str = " ".join(token_map)
    print(f"Full str: {token_map_str}")

    phrase_token_map = get_token_map(prompt=phrase, padding="do_not_pad")
    # Remove <bos> and <eos> in substr
    phrase_token_map = phrase_token_map[1:-1]
    phrase_token_map_len = len(phrase_token_map)
    phrase_token_map_str = " ".join(phrase_token_map)
    print(f"phrase str: {phrase_token_map_str}")

    # Count the number of token before substr
    # The substring comes with a trailing space that needs to be removed by minus one in the index.
    obj_first_index = len(token_map_str[: token_map_str.index(phrase_token_map_str) - 1].split(" "))

    obj_position = list(range(obj_first_index, obj_first_index + phrase_token_map_len))
    print(f"phrase_indices: {obj_position}")

    return obj_position


def expand_word_in_text(text, word, times=8):
    # Create the expanded word by repeating it the specified number of times, separated by spaces
    expanded_word = ' '.join([word] * times)
    
    # Replace the original word in the text with the expanded version
    return text.replace(word, expanded_word)


def compute_cross_attn_loss(cross_attn, obj_masks):
    '''
    Computes BCE loss between cross attention maps and resized object masks.

    Args:
    cross_attn: torch.Tensor of shape (batch_size, num_frames, height, width)
        The cross attention maps.
    obj_masks: torch.Tensor of shape (batch_size, num_frames, mask_height, mask_width)
        The object masks.

    Returns:
    torch.Tensor
        The computed loss.
    '''
    batch_size, num_frames, height, width = cross_attn.shape
    obj_masks = obj_masks.squeeze(1).to(dtype=cross_attn.dtype)
    resized_masks = F.interpolate(obj_masks, size=(height, width), mode='bilinear', align_corners=False)
    
    # Assuming that both cross_attn and resized_masks are probabilities (sigmoid applied if needed)
    # Calculate the BCE loss
    loss = F.binary_cross_entropy(cross_attn, resized_masks, reduction='mean')

    return loss

def save_attn_map(tensor, filename):
    """
    Save a tensor representing a grayscale image or video to a file using PIL.
    The tensor should have a shape (b, f, h, w) where:
    b: batch size
    f: number of frames (1 for a single image)
    h: height
    w: width
    
    Args:
    tensor (torch.Tensor): Input tensor.
    filename (str): Output filename, without file extension.
    """
    # Ensure tensor is on CPU and detach it from any computation graph
    tensor = tensor.cpu().detach().float()
    
    # Handle single image case
    if tensor.shape[1] == 1:
        # Squeeze to remove the frame dimension since it's 1
        image = tensor.squeeze(1)  # now shape should be (b, h, w)
        # Save each image in the batch
        for i, img in enumerate(image):
            # Convert to numpy array, ensure data type is uint8, and scale if necessary
            img = (img * 255).numpy().astype('uint8')
            # with np.printoptions(threshold=np.inf):
            #     print(img)
            np.set_printoptions()
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(img)
            pil_img.save(f"{filename}_{i}.jpg")
    else:
        # Handle video case
        for i, video in enumerate(tensor):
            # Prepare frames for the GIF
            frames = []
            for frame in video:
                # Convert to numpy array, ensure data type is uint8, and scale if necessary
                frame = (frame * 255).numpy().astype('uint8')
                # Convert numpy array to PIL Image and append to frames
                frames.append(Image.fromarray(frame))
            
            # Save as GIF
            frames[0].save(f"{filename}_{i}.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)


def save_transformed_images(transformed_tensors, filename):
    """
    Save the transformed tensor as an image file.
    """
    # # Define the transformation sequence
    # image_transforms = transforms.Compose([
    #     transforms.Resize((height, height), interpolation=transforms.InterpolationMode.BILINEAR),
    #     transforms.Pad(((width - height) // 2, 0, (width - height) // 2, 0)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5]),
    # ])
    
    # # Apply the transformations to the input image
    # transformed_tensor = image_transforms(input_image)
    
    # Denormalize the tensor
    mean = torch.tensor([0.5])
    std = torch.tensor([0.5])
    transformed_tensors = transformed_tensors * std[:, None, None] + mean[:, None, None]
    
    # Convert the tensor to a PIL image
    for i, img in enumerate(transformed_tensors):
        img = img.mul(255).clamp(0, 255).byte()
        img = Image.fromarray(img.numpy().transpose(1, 2, 0))
        img.save(f"{filename}_{i}.jpg")


def save_custom_weights(trained_state_dict, path):
    save_dict = {}
    if trained_state_dict is not None:
        trained_state_dict = {key: value.to(torch.float16) for key, value in trained_state_dict.items()}
        save_dict['unet_trained_weights'] = trained_state_dict
    torch.save(save_dict, path)


def load_custom_weights(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['unet_trained_weights'], strict=False)