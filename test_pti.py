from src.models.adapter.lora_utils import patch_pipe
from src.models import build_stable_diffusion
from diffusers import DiffusionPipeline
import hydra
import omegaconf
from hydra import compose, initialize
import torch


initialize(version_base='1.3', config_path='./configs') 
cfg = compose(config_name='PivotalTuningInversion')
def eval_resolver(s: str):
    return eval(s)
omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver)

sd_model = build_stable_diffusion(cfg.model)
device = "cuda:0"

vae = sd_model[0].to(device)

unet = sd_model[1].to(device)
noise_scheduler = sd_model[2]

if len(sd_model[3]) == 2:
    tokenizer1 = sd_model[3][0]
    tokenizer2 = sd_model[3][1]
    text_encoder1 = sd_model[4][0].to(device)
    text_encoder2 = sd_model[4][1].to(device)
else:
    tokenizer1 = sd_model[3]
    text_encoder1 = sd_model[4].to(device)

pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        text_encoder = text_encoder1,
        text_encoder_2 = text_encoder2,
        vae = vae,
        unet = unet,
        tokenizer = tokenizer1,
        tokenizer_2 = tokenizer2,
        torch_dtype = torch.float32,
        use_safetensors=True
        ).to(device)

patch_pipe(
    pipeline,
    "/home2/kkms4641/GenerativeModel_Tobigs_Conference_20-21/model_dumps/model/train1"
)

prompt = "<s1> headshot photo style, christmas background"

images = []
for i in range(10):
    image = pipeline(
        prompt=prompt,
        num_inference_steps=40,
        denoising_end=0.8,
        output_type="latent",
    ).images
    images.append(image)
    
pipeline.to('cpu')
torch.cuda.empty_cache()

refiner = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-refiner-1.0",
text_encoder_2=pipeline.text_encoder_2,
vae=pipeline.vae,
use_safetensors=True,
).to(device)


img_path = '/home2/kkms4641/GenerativeModel_Tobigs_Conference_20-21/model_dumps/vis'
#os.makedirs(img_path+cfg.exp_name,exist_ok=True)
for i in range(10):
    image = refiner(
        prompt=prompt,
        num_inference_steps=40,
        denoising_start=0.8,
        image=images[i],
    ).images[0]
    #image.save(f'{img_path}/{cfg.exp_name}/exp{i}.png','png')
    image.save(f'{img_path}/exp{i}.png','png')

