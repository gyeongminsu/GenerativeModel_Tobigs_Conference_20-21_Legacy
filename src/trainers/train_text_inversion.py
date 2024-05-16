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

class BaseTrainer():
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 logger,
                 sd_model):
        super().__init__()
        
        self.vae