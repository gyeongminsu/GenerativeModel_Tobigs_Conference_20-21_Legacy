from .train_text_inversion import TextualInversionTrainer
from .train_perform_tuning import PerformTuningTrainer
from dotmap import DotMap
from omegaconf import OmegaConf

def build_trainer(cfg,
                  device,
                  train_loader,
                  logger,
                  sd,
                  placeholder_ids):
    
    train_type = cfg.train_type

    OmegaConf.resolve(cfg)
    cfg = DotMap(OmegaConf.to_container(cfg))
    
    if train_type == 'TI':
        trainer = TextualInversionTrainer(cfg=cfg,
                       device=device,
                       train_loader=train_loader,
                       logger=logger,
                       sd_model=sd,
                       placeholder_ids=placeholder_ids)
    elif train_type == 'PTI':
        trainer = PerformTuningTrainer(
            cfg=cfg,
            device=device,
            train_loader=train_loader,
            logger=logger,
            sd_model=sd,
            placeholder_ids=placeholder_ids
        )

    return trainer