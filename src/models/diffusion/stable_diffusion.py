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
        pretrained_vae_name_or_path,
        torch_dtype =torch.float16
    )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
        variant='fp16',
        torch_dtype =torch.float16 if dtype=='fp16' else torch.float32,
        use_safetensors=True
    )
    
    return (
        vae,
        unet
    )
    