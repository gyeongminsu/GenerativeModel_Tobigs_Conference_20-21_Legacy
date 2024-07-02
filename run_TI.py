import argparse
from dotmap import DotMap
import torch
import wandb
import hydra
import omegaconf
from hydra import compose, initialize
import PIL
import os

from src.common.train_utils import set_global_seeds, add_placeholder_to_tokenizer, init_token_embeddings
from src.models import build_stable_diffusion
from src.trainers import build_trainer
from src.datasets import build_dataloader
from src.common.logger import WandbTrainerLogger

def run(args):

    args = DotMap(args)
    
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    
    initialize(version_base='1.3', config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)
    def eval_resolver(s: str):
        return eval(s)
    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver)
    
    set_global_seeds(cfg.seed)
    device = torch.device(cfg.device)


    sd = build_stable_diffusion(cfg.model)
    
    
    cfg.trainer.placeholder.init_token = args.init_token
    # TODO: Init token 두번째 토큰부터 random init할 수 있게 만들기
    sd[3][0], init_token_id, placeholder_ids = add_placeholder_to_tokenizer(sd[3][0],**cfg.trainer.placeholder)
    sd[3][0], sd[4][0] = init_token_embeddings(sd[3][0],sd[4][0],init_token_id, placeholder_ids)
    placeholder_token = " ".join(sd[3][0].convert_ids_to_tokens(placeholder_ids))
    
    train_loader = build_dataloader(cfg.dataset,sd[3][0],sd[3][1],placeholder_token=placeholder_token)

    
    logger = WandbTrainerLogger(cfg)
    
    trainer = build_trainer(cfg=cfg.trainer,device=device,logger=logger,sd=sd,train_loader=train_loader, placeholder_ids = placeholder_ids)

 
    pipeline = trainer.train()

    wandb.finish()
    #torch.save(unet.state_dict(), f"./data/weights/{args.exp_name}.pth")

    # Debug <s1>
    prompt = f"{placeholder_token} headshot photo style, christmas background"
    from diffusers import DiffusionPipeline
    images = []
    for i in range(10):
        image = pipeline(
            prompt=prompt,
            num_inference_steps=40,
            denoising_end=0.8,
            output_type="latent",
        ).images
        images.append(image)
        
    pipeline.to('cpu')
    torch.cuda.empty_cache()
    
    refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipeline.text_encoder_2,
    vae=pipeline.vae,
    use_safetensors=True,
    ).to(device)


    img_path = '/home2/kkms4641/GenerativeModel_Tobigs_Conference_20-21/'
    os.makedirs(img_path+cfg.exp_name,exist_ok=True)
    for i in range(10):
        image = refiner(
            prompt=f"{placeholder_token} headshot photo style, christmas background",
            num_inference_steps=40,
            denoising_start=0.8,
            image=images[i],
        ).images[0]
        image.save(f'{img_path}/{cfg.exp_name}/exp{i}.png','png')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs')
    parser.add_argument('--config_name', type=str, default='TextualInversion')
    parser.add_argument('--init_token', type=str, default='Dog')
    parser.add_argument('--overrides', action='append', default=[])
    args = parser.parse_args()
    
    run(vars(args))