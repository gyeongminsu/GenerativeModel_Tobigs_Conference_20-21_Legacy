from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

import torch

def get_diffusion_model(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    revision,
    dtype
):
    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_name_or_path or pretrained_model_name_or_path,
        subfolder = 'None' if pretrained_vae_name_or_path else 'vae',
        revision=None if pretrained_vae_name_or_path else revision,
        torch_dtype =torch.float16 if dtype=='fp16' else torch.float32
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
        torch_dtype =torch.float16 if dtype=='fp16' else torch.float32
    )
    
    return (
        vae,
        unet
    )
    