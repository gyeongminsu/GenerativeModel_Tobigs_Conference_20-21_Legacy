import numpy as np
import tqdm
import random
import wandb
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.common.schedulers import CosineAnnealingWarmUpRestarts
from src.common.train_utils import *

class BaseTrainer():
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 logger,
                 sd_model,
                 placeholder_token,
                 init_token,
                 num_vectors):
        super().__init__()
        
        self.vae = sd_model[0]
        self.unet = sd_model[1]
        self.noise_scheduler = sd_model[2]
        
        if len(sd_model[3]) == 2:
            self.tokenizer1 = sd_model[3][0]
            self.tokenizer2 = sd_model[3][1]
            self.text_encoder1 = sd_model[4][0]
            self.text_encoder2 = sd_model[4][1]
        else:
            self.tokenizer1 = sd_model[3]
            self.text_encoder1 = sd_model[4]
            
        self.tokenizer1, init_token_id, placeholder_ids = add_placeholder_to_tokenizer(self.tokenizer1,placeholder_token,init_token,num_vectors)
        
        self.text_encoder1.resize_token_embeddings(len(self.tokenizer1))
        
        token_embeds = self.text_encoder1.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id in placeholder_ids:
                token_embeds[token_id] = token_embeds[init_token_id].clone()
                
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        
        self.text_encoder_1.text_model.encoder.requires_grad_(False)
        self.text_encoder_1.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder_1.text_model.embeddings.position_embedding.requires_grad_(False)
        
        self.optimizer = self._build_optimizer(cfg.optimizer_type,cfg.optimizer)
        
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