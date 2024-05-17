from typing import List
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import torch

def get_clip_model(
    pretrained_model_name_or_path,
    is_sdxl,
    revision,
    dtype
):
    tokenizer = []
    text_encoder = []
    
    tokenizer_1 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer', dtype=torch.float16 if dtype=='fp16' else torch.float32)
    text_encoder_1 = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder='text_encoder',revision=revision,
        torch_dtype =torch.float16 if dtype=='fp16' else torch.float32,use_safetensors=True
    )
    
    tokenizer.append(tokenizer_1)
    text_encoder.append(text_encoder_1)
    
    if is_sdxl:
        tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2", dtype=torch.float16 if dtype=='fp16' else torch.float32)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2", revision=revision,
        torch_dtype =torch.float16 if dtype=='fp16' else torch.float32,use_safetensors=True
        )
        tokenizer.append(tokenizer_2)
        text_encoder.append(text_encoder_2)
        
    return tokenizer,text_encoder