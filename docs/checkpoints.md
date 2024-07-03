Please replace the lavie model's config file `checkpoints/lavie/unet/config.json`, `checkpoints/lavie/scheduler/scheduler_config.json`, and `checkpoints/lavie/model_index.json` to below.

Unet config.

``` json
{
  "_class_name": "LaVieModel",
  "_diffusers_version": "0.25.0",
  "act_fn": "silu",
  "attention_head_dim": 8,
  "block_out_channels": [
    320,
    640,
    1280,
    1280
  ],
  "center_input_sample": false,
  "class_embed_type": null,
  "cross_attention_dim": 768,
  "down_block_types": [
    "CrossAttnDownBlock3D",
    "CrossAttnDownBlock3D",
    "CrossAttnDownBlock3D",
    "DownBlock3D"
  ],
  "downsample_padding": 1,
  "dual_cross_attention": false,
  "flip_sin_to_cos": true,
  "freq_shift": 0,
  "in_channels": 4,
  "layers_per_block": 2,
  "mid_block_scale_factor": 1,
  "mid_block_type": "UNetMidBlock3DCrossAttn",
  "norm_eps": 1e-05,
  "norm_num_groups": 32,
  "num_class_embeds": null,
  "only_cross_attention": false,
  "out_channels": 4,
  "resnet_time_scale_shift": "default",
  "sample_size": 64,
  "up_block_types": [
    "UpBlock3D",
    "CrossAttnUpBlock3D",
    "CrossAttnUpBlock3D",
    "CrossAttnUpBlock3D"
  ],
  "upcast_attention": false,
  "use_first_frame": false,
  "use_linear_projection": false,
  "use_relative_position": false
}
```

Scheduler config.

``` json
{
  "_class_name": "DDPMScheduler",
  "_diffusers_version": "0.7.0.dev0",
  "beta_end": 0.02,
  "beta_schedule": "linear",
  "beta_start": 0.0001,
  "num_train_timesteps": 1000,
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "trained_betas": null,
  "clip_sample": false
}
```

model_index config.

``` json
{
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.2.2",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "DDPMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```