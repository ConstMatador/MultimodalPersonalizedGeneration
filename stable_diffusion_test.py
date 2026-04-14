import torch
from diffusers import StableDiffusionXLPipeline

model_dir = "/root/TOS/ZhongzhengWang/model/stable-diffusion-xl-base-1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_dir,
    torch_dtype=dtype,
    use_safetensors=True,
    variant="fp16" if dtype == torch.float16 else None,  # 尽量加载fp16权重
).to(device)

# 关键：保证 VAE 和主dtype一致（不要再用 float32）
pipe.vae.to(dtype=dtype, device=device)
pipe.unet.to(dtype=dtype, device=device)
pipe.text_encoder.to(dtype=dtype, device=device)
pipe.text_encoder_2.to(dtype=dtype, device=device)

prompt = "Movie poster of Toy Story, Woody and Buzz, cinematic, detailed"
negative_prompt = "low quality, blurry, watermark"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.0,
    height=1024,
    width=1024
).images[0]

image.save("out.png")
print("saved: out.png")
