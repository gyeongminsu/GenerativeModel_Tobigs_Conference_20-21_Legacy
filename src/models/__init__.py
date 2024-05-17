from .tokenizer import *
from .diffusion import *
from omegaconf import OmegaConf

def build_stable_diffusion(
    cfg
):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    diffusion_cfg = cfg['diffusion']
    tokenizer_cfg = cfg['tokenizer']
    
    vae, unet, noise_scheduler = get_ldm_model(**diffusion_cfg)
    tokenizer,text_encoder = get_tokenizer_model(**tokenizer_cfg)
    
    sd = (vae,
        unet,
        noise_scheduler,
        tokenizer,
        text_encoder)
    
    return sd