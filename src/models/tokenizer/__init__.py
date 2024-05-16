from .clip import *

def get_tokenizer_model(
    pretrained_model_name_or_path,
    tokenizer_type,
    is_sdxl,
    revision,
    dtype
):
    if tokenizer_type == 'clip':
        tokenizer, text_encoder = get_clip_model(pretrained_model_name_or_path,is_sdxl,revision,dtype)
    else:
        raise NotImplementedError
    
    return tokenizer, text_encoder