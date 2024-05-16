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
    
def add_placeholder_to_tokenizer(tokenizer,placeholder_token,init_token,num_vectors):
    placeholder_tokens = [placeholder_token]
    
    if num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {num_vectors}")
    
    additional_tokens = []
    
    for i in range(1, num_vectors):
        additional_tokens.append(f"{placeholder_token}_{i}")
    placeholder_tokens += additional_tokens
    
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    
    if num_added_tokens != num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
        
    token_ids = tokenizer.encode(init_token, add_special_tokens=False)
    
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")
    
    init_token_id = token_ids[0]
    placeholder_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    
    return tokenizer, init_token_id, placeholder_ids