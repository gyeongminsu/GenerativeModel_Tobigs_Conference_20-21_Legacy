from .diffusion_scheduler import *
from .stable_diffusion import *
from diffusers import DiffusionPipeline

def get_ldm_model(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    scheduler_type,
    revision,
    dtype
):
    
   
    vae, unet = get_diffusion_model(pretrained_model_name_or_path, pretrained_vae_name_or_path, revision,dtype)
    noise_scheduler = get_diffusion_scheduler(pretrained_model_name_or_path, scheduler_type)
    
    return vae, unet, noise_scheduler
    