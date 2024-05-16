import numpy as np
import random
import torch


def set_global_seeds(seed):
    torch.backends.cudnn.deterministic = True
    # https://huggingface.co/docs/diffusers/v0.9.0/en/optimization/fp16
    torch.backends.cudnn.benchmark = True 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)