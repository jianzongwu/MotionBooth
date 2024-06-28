import argparse

import torch

from src.models.lavie.unet import LaVieModel


def main(args):
    lavie_ckpt_path = 'checkpoints/lavie_base.pt'
    state_dict = torch.load(lavie_ckpt_path, map_location=lambda storage, loc: storage)["ema"]
    print(state_dict.keys())

    unet = LaVieModel.from_pretrained_2d(
        "checkpoints/stable-diffusion-v1-4", subfolder="unet"
    )
    unet.load_state_dict(state_dict)
    torch.save(unet.state_dict(), "checkpoints/stable-diffusion-v1-4/unet/diffusion_pytorch_model.bin")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
