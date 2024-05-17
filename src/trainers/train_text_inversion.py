import numpy as np
import tqdm
import random
import wandb
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from diffusers import DiffusionPipeline

from src.common.schedulers import CosineAnnealingWarmUpRestarts
from src.common.train_utils import *

class TextualInversionTrainer():
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 logger,
                 sd_model,
                 placeholder_ids
                 ):
        super().__init__()
        
        self.device = device
        self.cfg = cfg
        self.placeholder_ids = placeholder_ids
        
        self.vae = sd_model[0].to(self.device)
        
        self.unet = sd_model[1].to(self.device)
        self.noise_scheduler = sd_model[2]
        self.train_loader = train_loader
        
        if len(sd_model[3]) == 2:
            self.tokenizer1 = sd_model[3][0]
            self.tokenizer2 = sd_model[3][1]
            self.text_encoder1 = sd_model[4][0].to(self.device)
            self.text_encoder2 = sd_model[4][1].to(self.device)
        else:
            self.tokenizer1 = sd_model[3]
            self.text_encoder1 = sd_model[4].to(self.device)
            
        self.logger = logger
        
        self._freeze_wo_embeddings()
        self.optimizer = self._build_optimizer(cfg.optimizer_type,cfg.optimizer)
        cfg.scheduler.T_0 = len(train_loader)
        self.scheduler = self._build_scheduler(self.optimizer,cfg.scheduler)
        
        self.dtype = torch.float16
        if cfg.dtype == 'fp32':
            self.dtype = torch.float32
        
        self.epoch = 0
        self.step = 0
        
    def _build_optimizer(self, optimizer_type, optimizer_cfg):
        if optimizer_type == 'adamw':
            return optim.AdamW(self.text_encoder1.get_input_embeddings().parameters(), **optimizer_cfg)
        elif optimizer_type == 'adam':
            return optim.Adam(self.text_encoder1.get_input_embeddings().parameters(), **optimizer_cfg)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.text_encoder1.get_input_embeddings().parameters(), **optimizer_cfg)
        else:
            raise NotImplementedError
        
    def _build_scheduler(self, optimizer, scheduler_cfg):
        return CosineAnnealingWarmUpRestarts(optimizer=optimizer, **scheduler_cfg)
    
    def _freeze_wo_embeddings(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder2.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        self.text_encoder1.text_model.encoder.requires_grad_(False)
        self.text_encoder1.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder1.text_model.embeddings.position_embedding.requires_grad_(False)
        
    def train(self):
        cfg = self.cfg
        num_epochs = cfg.epochs
        loss_type = cfg.loss
        self.logger.log_to_wandb(self.step)
        
        original_embeds_params = self.text_encoder1.get_input_embeddings().weight.data.clone()
        
        for epoch in range(num_epochs):
            self.text_encoder1.train()
            print(f'Epoch: {epoch}')
            for step, batch in enumerate(tqdm.tqdm(self.train_loader)):
                
                train_logs = {}
                
                img = batch['pixel_values'].to(self.device,dtype=self.dtype)
                
                # convert images to latent space
                
                latents = self.vae.encode(img).latent_dist.sample().detach()

                latents = latents * self.vae.config.scaling_factor
                # smaple noise that we'll add to the latents
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,(batch_size,),device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = self.noise_scheduler.add_noise(latents,noise,timesteps)
                
                # text embedding
                encoder_hidden_states_1 = (
                    self.text_encoder1(batch['input_ids_1'].to(self.device), output_hidden_states=True).hidden_states[-2].to(dtype=self.dtype)
                )

                encoder_output_2 = self.text_encoder2(
                batch["input_ids_2"].reshape(batch["input_ids_1"].shape[0], -1).to(self.device), output_hidden_states=True
                )
                encoder_hidden_states_2 = encoder_output_2.hidden_states[-2].to(dtype=self.dtype)

                original_size = [
                    (batch["original_size"][0][i].item(), batch["original_size"][1][i].item())
                    for i in range(batch_size)
                ]

                crop_top_left = [
                    (batch["crop_top_left"][0][i].item(), batch["crop_top_left"][1][i].item())
                    for i in range(batch_size)
                ]
                target_size = (cfg.resolution, cfg.resolution)
                add_time_ids = torch.cat(
                    [
                        torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                        for i in range(batch_size)
                    ]
                ).to(self.device, dtype=self.dtype)
                added_cond_kwargs = {"text_embeds": encoder_output_2[0], "time_ids": add_time_ids}
                encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)
                
                # Predict the noise residual

                model_pred = self.unet(
                    noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                ).sample

                # Get the target for loss depending on the prediction type
                if self.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                train_logs['loss'] = loss.item()
                train_logs['lr'] = self.scheduler.get_lr()[0]
                self.logger.update_log(**train_logs)
                
                if self.step % cfg.log_every == 0:
                    self.logger.log_to_wandb(self.step)
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # update only new token
                index_no_updates = torch.ones((len(self.tokenizer1),), dtype=torch.bool)
                index_no_updates[min(self.placeholder_ids) : max(self.placeholder_ids) + 1] = False
                
                with torch.no_grad():
                    self.text_encoder1.get_input_embeddings().weight[
                        index_no_updates
                    ] = original_embeds_params[index_no_updates]
                
                self.step += 1
            self.epoch += 1

        # after training, make PipeLine
        pipeline = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                text_encoder = self.text_encoder1,
                text_encoder_2 = self.text_encoder2,
                vae = self.vae,
                unet = self.unet,
                tokenizer = self.tokenizer1,
                tokenizer_2 = self.tokenizer2,
                torch_dtype = torch.float32,
                use_safetensors=True
                ).to(self.device)
        
        return pipeline