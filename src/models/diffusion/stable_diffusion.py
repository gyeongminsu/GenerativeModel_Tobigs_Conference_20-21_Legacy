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
    if dtype == 'fp16':
        vae = AutoencoderKL.from_pretrained(
            pretrained_vae_name_or_path,
            torch_dtype = torch.float16
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder='vae',
            torch_dtype = torch.float32,
            use_safetensors=True
        )

    if dtype == 'fp16':
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            variant=dtype,
            torch_dtype =torch.float16 if dtype=='fp16' else torch.float32,
            use_safetensors=True
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            torch_dtype =torch.float16 if dtype=='fp16' else torch.float32,
            use_safetensors=True
        )
    return (
        vae,
        unet
    )
    