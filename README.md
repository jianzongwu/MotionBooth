<br />
<p align="center">
  <h1 align="center">MotionBooth: Motion-Aware Customized <br> Text-to-Video Generation</h1>
  <p align="center">
    <br />
    <a href="https://jianzongwu.github.io/"><strong>Jianzong Wu</strong></a>
    ·
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li</strong></a>
    ·
    <a href="https://zengyh1900.github.io/"><strong>Yanhong Zeng</strong></a>
    ·
    <a href="https://zhangzjn.github.io/"><strong>Jiangning Zhang</strong></a>
    .
    <a href="https://qianyuzqy.github.io/"><strong>Qianyu Zhou</strong></a>
    .
    <a href="https://github.com/ly015"><strong>Yining Li</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=T4gqdPkAAAAJ"><strong>Yunhai Tong</strong></a>
    .
    <a href="https://chenkai.site/"><strong>Kai Chen</strong></a>
  </p>

  <p align="center">
    <a href='https://arxiv.org/abs/2406.17758'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://github.com/jianzongwu/MotionBooth'>
      <img src='https://img.shields.io/badge/Github-Code-blue?style=flat&logo=Github' alt='Code'>
    </a>
    <a href='https://jianzongwu.github.io/projects/motionbooth'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=webpack' alt='Project Page'>
    </a>
  </p>
<br />

## Examples

**Customization and subject motion control**

<img src="assets/demo1.gif" width="600">

**Hybrid control on customization, subject and camera motion**

<img src="assets/demo2.gif" width="600">


## 🎉 News

- [2024-6-28] Inference code, training code, and checkpoints are released!

## 📖 Abstract

In this work, we present MotionBooth, an innovative framework designed for animating customized subjects with precise control over both object and camera movements. By leveraging a few images of a specific object, we efficiently fine-tune a text-to-video model to capture the object's shape and attributes accurately. Our approach presents subject region loss and video preservation loss to enhance the subject's learning performance, along with a subject token cross-attention loss to integrate the customized subject with motion control signals. Additionally, we propose training-free techniques for managing subject and camera motions during inference. In particular, we utilize cross-attention map manipulation to govern subject motion and introduce a novel latent shift module for camera movement control as well. MotionBooth excels in preserving the appearance of subjects while simultaneously controlling the motions in generated videos. Extensive quantitative and qualitative evaluations demonstrate the superiority and effectiveness of our method. Models and codes will be made publicly available.

## 🛠️ Quick Start

### Installation

- In this repo, we use Python 3.11 and PyTorch 2.1.2. Newer versions of Python and PyTorch may be also compatible.

``` bash
# Create a new environment with Conda
conda create -n motionbooth python=3.11
conda activate motionbooth
# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

- We strongly recommend using [Diffusers](https://huggingface.co/docs/diffusers/index) to be the codebase for the training and inference of diffusion-based models. Diffusers provides easy and compatible implementations around diffusion-based generative models.
- In this repo, we use diffusers 0.29.0, if you use newer or older versions, you may need to adjust some import function paths manually. Please refer to the [Diffusers document](https://huggingface.co/docs/diffusers/index) for details.

``` bash
# Install diffusers, transformers, and accelerate
conda install -c conda-forge diffusers==0.29.0 transformers accelerate
# Install xformers for PyTorch 2.1.2
pip install xformers==0.0.23.post1
# Install other dependencies
pip install -r requirements.txt
```

### Data Preparation

We collect 26 objects from [DreamBooth](https://dreambooth.github.io/) and [CustomDiffusion](https://github.com/adobe-research/custom-diffusion) to perform the experiments in paper. These objects include pets, plushies, toys, cartoons, and vehicles. We also annotate masks for each image. We name it the MotionBooth dataset. Please download our dataset from [huggingface](https://huggingface.co/datasets/jianzongwu/MotionBooth).

Note that a few images from the original datasets are deleted because the low quality of the obtained masks. Additionally, a few images are resized and cropped to square shapes.

After downloading, please unzip and place the dataset under the `data` folder. It should look like this:

```
data
  |- MotionBooth
    |- images
      |- cat2
      |- ...
    |- masks
      |- cat2
      |- ...
  |- scripts
```

### Pre-trained Model Preparation

We use [Zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w) and [LaVie-base](https://huggingface.co/Vchitect/LaVie) for the base T2V models. Please download Zeroscope from the [official huggingface page](https://huggingface.co/cerspense/zeroscope_v2_576w). For LaVie, we provide a script to convert their original checkpoint into the format that is suitable for Diffusers. Please download the [LaVie-base](https://huggingface.co/Vchitect/LaVie) model and the [Stable-Diffusion-v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) checkpoint.

Then, organize the pre-trained models in the `checkpoints` folder.

```
checkpoints
  |- zeroscope_v2_576w
  |- stable-diffusion-v1-4
  |- lavie_base.pt
```

Then, run the following command to convert the checkpoint

``` bash
python -m scripts.convert_ckpts.convert_lavie_to_diffusers
``` 

Then, rename the `stable-diffusion-v1-4` folder to `lavie`. Additionally, you should replace the config file to LaVie's configs, following [checkpoint guide](docs/checkpoints.md).

The final checkpoint folder looks like this:

```
checkpoints
  |- zeroscope_v2_576w
  |- lavie
  |- lavie_base.pt (Not used anymore)
```

We use the converted lavie model for all the experiments.


### Inference

For quick inference and re-producing the examples in paper, please download our trained customized checkpoints for the target subjects in [huggingface](https://huggingface.co/jianzongwu/MotionBooth). The names of the checkpoints correspond to the subject names in the MotionBooth dataset.

Please place the checkpoints in che `checkpoints` folder like this:

```
checkpoints
  |- customized
    |- zeroscope
      |- ...
    |- lavie
      |- ...
  |- zeroscope_v2_576w
  |- lavie
  |- lavie_base.pt (Not used anymore)
```

We use simple script files to indicate the subject and camera motion. We provide several examples in `data/scripts`. In these script files, the "bbox" controls the bounding box sequence for the subjects' motion, while the "camera speed" controls the corresponding camera motion speed.

We provide the inference script in `scripts/inference.py` for all types of MotionBooth applications. It uses [Accelerate PartialState](https://huggingface.co/docs/accelerate/index) to support multi GPU inference.

#### [Vanilla T2V] Control the camera motion without customized subjects

The latent shift module proposed in paper can control the camera motion freely whether or not with a customized model. We provide scripts in `data/scripts/camera` to control the camera motion in vanilla text-to-video pipelines.

``` bash
python -m scripts.inference \
    --script_path data/scripts/camera/waterfall.json \
    --model_name lavie \
    --num_samples 1 \
    --start_shift_step 10 \
    --max_shift_steps 10
```

You can check the meaning of each parameter in the bottom of the script file.

For multi GPU inference, please run commands like this:

``` bash
accelerate launch \
    --multi_gpu \
    -m scripts.inference \
    --script_path data/scripts/camera/waterfall.json \
    --model_name lavie \
    --num_samples 8 \
    --start_shift_step 10 \
    --max_shift_steps 10
```

Feel free to try other scripts in `data/scripts/camera` and your own text prompts or camera speeds!

#### [Customized T2V] Control the camera motion

By loading the checkpoint fine-tuned on a specific subject, our latent shift module can control the camera motion of the generated videos when depicting the given subject.

``` bash
python -m scripts.inference \
    --script_path data/scripts/customized_camera/run_grass.json \
    --model_name lavie \
    --customize_ckpt_path checkpoints/customized/lavie/plushie_panda.pth \
    --class_name "plushie panda" \
    --num_samples 1 \
    --start_shift_step 10 \
    --max_shift_steps 10
```

#### [Customized T2V] Control the subject motion

The subject motion control can also be complished with minimal computational and time cost added.

``` bash
python -m scripts.inference \
    --script_path data/scripts/customized_subject/jump_stairs.json \
    --model_name zeroscope \
    --customize_ckpt_path checkpoints/customized/zeroscope/pet_cat1.pth \
    --class_name cat \
    --num_samples 1 \
    --edit_scale 7.5 \
    --max_amp_steps 5
```

#### [Customized T2V] Control both the camera and subject motion

MotionBooth can also control both the camera and subject motion

``` bash
python -m scripts.inference \
    --script_path data/scripts/customized_both/swim_coral.json \
    --model_name lavie \
    --customize_ckpt_path checkpoints/customized/lavie/plushie_happysad.pth \
    --class_name "plushie happysad" \
    --num_samples 1 \
    --edit_scale 10.0 \
    --max_amp_steps 15 \
    --start_shift_step 10 \
    --max_shift_steps 10 \
    --base_seed 5
```

## Train

### Download Preservation Video Data

Note: An 80G memory GPU is needed for training on 24-frame video data!

Before training MotionBooth, please download video-text pair data from [Panda-70M](https://github.com/snap-research/Panda-70M).

Please first download the [panda70m_training_2m.csv](https://drive.google.com/file/d/1jWTNGjb-hkKiPHXIbEA5CnFwjhA-Fq_Q/view) from Panda-70M official release and place it into `data/panda/panda70m_training_2m.csv`.

To download random videos from the training set, we provide an easy-to-use [downloading script](scripts/download_dataset/panda.py) for downloading and organizing the videos from YouTube.

``` bash
python -m scripts.download_dataset.panda70m
```

After downloading, your `data` folder should look like this:

```
data
  |- MotionBooth
  |- scripts
  |- panda
    |- random_500
      |- {video1}
      |- {video2}
      |- ...
    |- captions_random.json
    |- data/panda/panda70m_training_2m.csv
```

### Train MotionBooth

The training procedure is as simple as running `scripts/train.py`. This is an example training LaVie on "dog3" in the MotionBooth dataset.

``` bash
python -m scripts.train \
    --config_path configs/lavie.yaml \
    --obj_name dog3
```

For tuning Zeroscope/LaVie for 300 steps, it takes you less than 20 minutes.

After the training is completed, you can place the saved checkpoints in the `logs` folder to `checkpoints/customized/` and run the inference!

And of course, you can prepare your own object and save the images and masks just like MotionBooth dataset.


## 📢 Disclaimer

Our framework is the first that is capable of generating diverse videos by taking any combination of customized subjects, subject motions, and camera movements as input. However, due to the variaity of generative video prior, the success rate is not guaranteed. Be patient and generate more samples under different random seeds to have better results. 🤗

## Citation

```
article{wu2024motionbooth,
  title={MotionBooth: Motion-Aware Customized Text-to-Video Generation},
  author={Jianzong Wu and Xiangtai Li and Yanhong Zeng and Jiangning Zhang and Qianyu Zhou and Yining Li and Yunhai Tong and Kai Chen},
  journal={NeurIPS},
  year={2024},
}
```
