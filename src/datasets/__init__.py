from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .textulainversion import TextualInversionDataset
from .dreambooth import DreamBoothDataset

def build_dataloader(cfg,tokenizer1,tokenizer2,placeholder_token):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    dataset_type = cfg['type']
    train_dataset_cfg = cfg['train_dataset']
        
    train_dataloader_cfg = cfg['train_dataloader']
    test_dataloader_cfg = cfg['test_dataloader']
    
    if dataset_type in ['TextualInversion']:
    
        train_dataset = TextualInversionDataset(tokenizer_1=tokenizer1,tokenizer_2=tokenizer2,placeholder_token=placeholder_token,**train_dataset_cfg)
        #test_dataset = load_imagenet1k(is_train=False)
    elif dataset_type in ['DreamBooth']:
        
        train_dataset = DreamBoothDataset(tokenizer_1=tokenizer1,tokenizer_2=tokenizer2,placeholder_token=placeholder_token,class_prompt=f"a photo of a cool {placeholder_token}",class_data_root='/home/shu/Desktop/Yongjin/GenAI/Project/GenerativeModel_Tobigs_Conference_20-21/data/dog2',**train_dataset_cfg)
        #test_dataset = load_imagenet1k(is_train=False)
    else:
        raise NotImplementedError
    
    train_loader = DataLoader(train_dataset, **train_dataloader_cfg, shuffle=True)
    #test_loader = DataLoader(test_dataset,**test_dataloader_cfg, shuffle=False)
    return train_loader #, test_loader