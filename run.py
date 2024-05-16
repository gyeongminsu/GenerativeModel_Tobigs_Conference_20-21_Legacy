from src.models import build_stable_diffusion
sd = build_stable_diffusion("stabilityai/stable-diffusion-xl-base-1.0",None,'clip','euler_discrete',True,None,'cuda:0')
print(sd)