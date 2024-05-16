from .tokenizer import *
from .diffusion import *
from src.common.train_utils import add_placeholder_to_tokenizer
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

def prepare_text_inversion(
    sd_model,
    placeholder_token,
    init_token,
    num_vectors  
):
    vae = sd_model[0]
    unet = sd_model[1]
    noise_scheduler = sd_model[2]
    
    if len(sd_model[3]) == 2:
        tokenizer1 = sd_model[3][0]
        tokenizer2 = sd_model[3][1]
        text_encoder1 = sd_model[4][0]
        text_encoder2 = sd_model[4][1]
    else:
        tokenizer1 = sd_model[3]
        text_encoder1 = sd_model[4]
    tokenizer1, init_token_id, placeholder_ids = add_placeholder_to_tokenizer(tokenizer1,placeholder_token,init_token,num_vectors)
        
    text_encoder1.resize_token_embeddings(len(tokenizer1))
    
    token_embeds = text_encoder1.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_ids:
            token_embeds[token_id] = token_embeds[init_token_id].clone()
            
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    
    text_encoder1.text_model.encoder.requires_grad_(False)
    text_encoder1.text_model.final_layer_norm.requires_grad_(False)
    text_encoder1.text_model.embeddings.position_embedding.requires_grad_(False)
    
    return (vae, unet, noise_scheduler, [tokenizer1,tokenizer2], [text_encoder1,text_encoder2])