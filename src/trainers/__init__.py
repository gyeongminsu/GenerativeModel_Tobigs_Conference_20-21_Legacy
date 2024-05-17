from .train_text_inversion import TextualInversionTrainer
from dotmap import DotMap
from omegaconf import OmegaConf

def build_trainer(cfg,
                  device,
                  train_loader,
                  logger,
                  sd,
                  placeholder_ids):
    
    OmegaConf.resolve(cfg)
    cfg = DotMap(OmegaConf.to_container(cfg))
    
    return TextualInversionTrainer(cfg=cfg,
                       device=device,
                       train_loader=train_loader,
                       logger=logger,
                       sd_model=sd,
                       placeholder_ids=placeholder_ids)