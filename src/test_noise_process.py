import torch
from flux_lora import TextImageDataset
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def decode_and_display_latent(cur_latent, prev_latent, noise, vae):
    # Unpack latents using FLUX pipeline's method
    batch_size, num_patches, channels = cur_latent.shape
    
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2
    height = int(np.sqrt(num_patches)) * 2  # Multiply by 2 since we packed 2x2 patches
    width = height
    
    # Unpack current latents
    cur_latent = cur_latent.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    cur_latent = cur_latent.permute(0, 3, 1, 4, 2, 5)
    cur_latent = cur_latent.reshape(batch_size, channels // 4, height, width)
    
    # Unpack previous latents
    prev_latent = prev_latent.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    prev_latent = prev_latent.permute(0, 3, 1, 4, 2, 5)
    prev_latent = prev_latent.reshape(batch_size, channels // 4, height, width)

    # Unpack noise the same way
    noise = noise.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    noise = noise.permute(0, 3, 1, 4, 2, 5)
    noise = noise.reshape(batch_size, channels // 4, height, width)
    
    # Scale the latents back to VAE range
    cur_latent_scaled = (cur_latent / vae.config.scaling_factor) + vae.config.shift_factor
    prev_latent_scaled = (prev_latent / vae.config.scaling_factor) + vae.config.shift_factor
    
    # Decode latents to images
    with torch.no_grad():
        cur_image = vae.decode(cur_latent_scaled).sample
        prev_image = vae.decode(prev_latent_scaled).sample
    
    # Convert to display format
    cur_image = (cur_image / 2 + 0.5).clamp(0, 1)
    prev_image = (prev_image / 2 + 0.5).clamp(0, 1)
    
    # Display images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(cur_image[0].permute(1, 2, 0).cpu().numpy())
    ax1.set_title('Current Timestep')
    ax1.axis('off')
    
    ax2.imshow(prev_image[0].permute(1, 2, 0).cpu().numpy())
    ax2.set_title('Previous Timestep')
    ax2.axis('off')

    # Display noise - take mean across channels for visualization
    noise_viz = noise[0].mean(dim=0).cpu().numpy()
    noise_viz = (noise_viz - noise_viz.min()) / (noise_viz.max() - noise_viz.min())
    ax3.imshow(noise_viz, cmap='viridis')
    ax3.set_title('Noise (Channel Mean)')
    ax3.axis('off')
    
    plt.show()

def test_noise_process():
    # Initialize pipeline components
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float32,
    )
    
    # Get components needed for testing
    vae = pipeline.vae
    tokenizer = pipeline.tokenizer
    tokenizer_2 = pipeline.tokenizer_2
    noise_scheduler = pipeline.scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move models to device and set to eval mode
    vae = vae.to(device)
    vae.eval()
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize(768, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(768),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create dataset and get one image
    dataset = TextImageDataset(
        image_dir="/home/ubuntu/Sam/invoke-training/src/paul_mescal_training/training_images",
        prompt_file="/home/ubuntu/Sam/invoke-training/src/paul_mescal_training/prompts.json",
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        transforms=transform
    )
    
    # Get first image
    batch = dataset[0]
    image = batch["images"].unsqueeze(0).to(device)  # Add batch dimension
    
    # Encode image to latent space
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample() * 0.18215
    
    # Pack latents
    batch_size, num_channels, height, width = latents.shape
    latents = pipeline._pack_latents(latents, batch_size, num_channels, height, width)
    
    # Sample timesteps at different points in the diffusion process
    timesteps = [100, 500, 900]  # Beginning, middle, and end of diffusion
    
    for t in timesteps:
        print(f"\nVisualizing noise at timestep {t}")
        timestep = torch.tensor([t], device=device).long()
        prev_timestep = torch.tensor([t-1], device=device).long()
        
        # Generate noise
        torch.manual_seed(42)  # For reproducibility
        noise = torch.randn_like(latents)
        
        # Get noised latents at both timesteps
        noisy_latents = noise_scheduler.scale_noise(
            sample=latents,
            timestep=timestep,
            noise=noise
        )
        
        prev_noisy_latents = noise_scheduler.scale_noise(
            sample=latents,
            timestep=prev_timestep,
            noise=noise
        )
        
        # Display results
        decode_and_display_latent(noisy_latents, prev_noisy_latents, noise, vae)
    
    # Display the original image
    plt.figure(figsize=(5, 5))
    plt.imshow(image[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_noise_process() 