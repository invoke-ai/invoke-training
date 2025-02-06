from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel
import torch 

def main():
    sdxl_id = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_id = "madebyollin/sdxl-vae-fp16-fix"
    # unet_id = "mapo-t2i/mapo-beta"

    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
    # unet = UNet2DConditionModel.from_pretrained(unet_id, torch_dtype=torch.float16)
    pipeline = DiffusionPipeline.from_pretrained(sdxl_id, vae=vae, torch_dtype=torch.float16).to("cuda")
    pipeline.load_lora_weights("/mnt/data-1/ryan-scratch/mapo_training_output/1738862170.202609/checkpoint-600", weight_name="pytorch_lora_weights.safetensors", adapter_name="cartoon_lora")

    # prompt = "A lion with eagle wings coming out of the sea , digital Art, Greg rutkowski, Trending artstation, cinematographic, hyperrealistic"
    prompt = "A lion with eagle wings coming out of the sea"
    image = pipeline(prompt=prompt, num_inference_steps=30, cross_attention_kwargs={"scale": 1.0}, generator=torch.manual_seed(0)).images[0]
    out_path = "output_1.png"
    image.save(out_path)
    print(f"Saved image to {out_path}")


if __name__ == "__main__":
    main()
