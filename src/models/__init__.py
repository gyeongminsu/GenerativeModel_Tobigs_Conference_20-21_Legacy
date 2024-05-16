from .tokenizer import *
from .diffusion import *

def build_stable_diffusion(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    tokenizer_type,
    scheduler_type,
    is_sdxl,
    revision,
    dtype,
    device
):
    vae, unet, noise_scheduler = get_ldm_model(
        pretrained_model_name_or_path,
        pretrained_vae_name_or_path,
        scheduler_type,
        revision,
        dtype
    )
    
    tokenizer,text_encoder = get_tokenizer_model(
        pretrained_model_name_or_path,
        tokenizer_type,
        is_sdxl,
        revision,
        dtype
    )
    
    text_encoder = [encoder.to(device) for encoder in text_encoder]
    sd = (vae.to(device),
        unet.to(device),
        noise_scheduler,
        tokenizer,
        text_encoder)
    
    return sd