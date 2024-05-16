from diffusers import DDPMScheduler, EulerDiscreteScheduler

def get_diffusion_scheduler(
    pretrained_model_name_or_path,
    scheduler_type
):
    if scheduler_type == 'ddpm':
        noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    elif scheduler_type == 'euler_discrete':
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder='scheduler')
    else:
        raise NotImplementedError
        
    return noise_scheduler