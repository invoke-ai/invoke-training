import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import types
from typing import List, Optional, Union, Dict, Any
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from accelerate import Accelerator
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
# import wandb


# Custom dataset for text-to-image training
class TextImageDataset(Dataset):
    def __init__(self, image_dir, prompt_file, tokenizer, tokenizer_2, transforms):
        self.image_dir = image_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Load prompts        
        with open(prompt_file, 'r') as f:
            self.prompts = json.load(f)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        
        # Get prompt for this image
        img_name = os.path.basename(image_path)
        prompt = self.prompts.get(img_name, "")
        
        # CLIP tokenization
        clip_tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # T5 tokenization
        t5_tokens = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,  # Max sequence length for T5
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "images": image,
            "clip_input_ids": clip_tokens.input_ids[0],
            "clip_attention_mask": clip_tokens.attention_mask[0],
            "t5_input_ids": t5_tokens.input_ids[0],
            "t5_attention_mask": t5_tokens.attention_mask[0],
            "prompt": prompt
        }


# Helper function to freeze params except LoRA
def freeze_params(model):
    """
    Freezes all parameters in a PyTorch model except for LoRA parameters.
    """
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
            
    # Print statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Frozen parameters: {total_params - trainable_params:,} ({100 * (total_params - trainable_params) / total_params:.2f}%)")


def parse_args(default=True):
    if default:
        return get_default_args()
    parser = argparse.ArgumentParser(description="FLUX LoRA Training Script")
    parser.add_argument("--pretrained_model_name", type=str, required=True, help="Path to pretrained FLUX model")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts for images")
    parser.add_argument("--output_dir", type=str, default="flux_lora_output", help="Output directory for saving model")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam optimizer")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision training")
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank for LoRA adaptation")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def get_default_args():
    args = argparse.Namespace()
    
    # Model paths
    args.pretrained_model_name = "black-forest-labs/FLUX.1-dev"
    args.output_dir = "flux_lora_output"


    # Training Data
    args.train_data_dir="/home/ubuntu/Sam/invoke-training/src/paul_mescal_training/training_images"
    args.prompt_file="/home/ubuntu/Sam/invoke-training/src/paul_mescal_training/prompts.json"
    
    # Training parameters
    args.resolution = 768  # Higher resolution for better quality
    args.train_batch_size = 1  # FLUX is memory-intensive
    args.learning_rate = 5e-5  # Slightly lower learning rate for stability
    args.max_train_steps = 2000  # More steps for better convergence
    args.gradient_accumulation_steps = 4  # For effective batch size of 4
    args.use_8bit_adam = True  # Save memory with 8-bit optimizer
    args.mixed_precision = "fp16"  # Use mixed precision to save memory
    
    # LoRA parameters
    args.lora_rank = 32  # Higher rank for better adaptation capacity
    args.lora_alpha = 64  # Alpha = 2 * rank is a good rule of thumb
    args.lora_dropout = 0.05  # Standard dropout value for LoRA
    
    # Miscellaneous
    args.seed = 42  # Standard random seed
    
    return args
    
def train():
    args = parse_args(default=True)
    print(args)
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize accelerator with gradient scaler for mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=True  # Direct parameter instead of using AcceleratorKwargs
    )
    
    # Load FLUX pipeline
    print("Loading FLUX pipeline...")
    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )

    # Extract components from the pipeline
    transformer = pipeline.transformer
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    tokenizer = pipeline.tokenizer
    tokenizer_2 = pipeline.tokenizer_2
    noise_scheduler = pipeline.scheduler
    
    # Configure LoRA for the transformer
    target_modules = [
        "to_q",  # Query projection
        "to_k",  # Key projection
        "to_v",  # Value projection
        "to_out.0",  # Output projection
        "ff.net.0.proj",  # MLP first projection
        "ff.net.2",  # MLP second projection
    ]
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        # bias="none",
        # task_type=TaskType.FEATURE_EXTRACTION
    )
    
    # Apply LoRA to transformer model
    print("Applying LoRA to transformer...")
    transformer = get_peft_model(transformer, lora_config)

    # Freeze other components and non-LoRA parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    freeze_params(transformer)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create dataset and dataloader
    dataset = TextImageDataset(
        args.train_data_dir,
        args.prompt_file,
        tokenizer,
        tokenizer_2,
        transform
    )
    print("DATASET CREATED")
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4
    )
    print("DATASET LOADER CREATED")

    # Optimizer
    print("INITIALIZING OPTIMIZER")
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(transformer.parameters(), lr=args.learning_rate)
        except ImportError:
            print("bitsandbytes not found. Using regular AdamW.")
            optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)
    
    # Prepare for accelerator
    print("PREPARE TRANSFORMER")
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    
    # Move models to device and set data types
    device = accelerator.device
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    
    # Move models to device and set dtype
    vae = vae.to(device).to(dtype)
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)
    
    # Training loop
    print("BEGIN INIT TRAINING ")
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), desc="Training")
    
    # Set models to eval mode
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print("BEGIN TRAINING LOOP ")
    print(type(pipeline.transformer))
    print(type(transformer))
    while global_step < args.max_train_steps:
        transformer.train()
        print(f"global step: {global_step}")
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                # Get image and text inputs
                images = batch["images"].to(device)
                clip_input_ids = batch["clip_input_ids"].to(device)
                clip_attention_mask = batch["clip_attention_mask"].to(device)
                t5_input_ids = batch["t5_input_ids"].to(device)
                t5_attention_mask = batch["t5_attention_mask"].to(device)
                
                # Encode text inputs
                with torch.no_grad():
                    # Encode with CLIP
                    clip_outputs = text_encoder(
                        clip_input_ids,
                        attention_mask=clip_attention_mask,
                        output_hidden_states=False
                    )
                    pooled_prompt_embeds = clip_outputs.pooler_output
                    
                    # Encode with T5
                    t5_outputs = text_encoder_2(
                        t5_input_ids,
                        attention_mask=t5_attention_mask,
                        output_hidden_states=False
                    )[0]
                    prompt_embeds = t5_outputs
                    
                    # Prepare text IDs
                    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device)
                    
                    # Encode images to latent space - match VAE dtype
                    images = images.to(device=device, dtype=dtype)
                    latents = vae.encode(images).latent_dist.sample() * 0.18215
                    
                    # Convert latents to float32 for training after VAE encoding
                    latents = latents.to(dtype=torch.float32)

                    # Pack latents
                    batch_size, num_channels, height, width = latents.shape
                    latents = pipeline._pack_latents(latents, batch_size, num_channels, height, width)
                    
                    # Prepare latent image IDs
                    latent_image_ids = pipeline._prepare_latent_image_ids(
                        batch_size, height // 2, width // 2, device, latents.dtype
                    )

                # Get noised latents - ensure all tensors are float32
                noisy_latents, prev_latents, noise, timestep = get_noised_latent_at_random_timestep(
                    sample_latents=latents,  # Already float32
                    scheduler=noise_scheduler
                )
                
                # Convert all inputs to float32
                noisy_latents = noisy_latents.to(dtype=torch.float32)
                prev_latents = prev_latents.to(dtype=torch.float32)
                noise = noise.to(dtype=torch.float32)
                timestep = timestep.to(dtype=torch.int64)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.float32)
                prompt_embeds = prompt_embeds.to(dtype=torch.float32)
                text_ids = text_ids.to(dtype=torch.float32)
                latent_image_ids = latent_image_ids.to(dtype=torch.float32)
                guidance = torch.full([1], 1.0, device=device, dtype=torch.float32)
                guidance = guidance.expand(latents.shape[0])

                # Forward pass
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timestep,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    guidance=guidance,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False
                )[0]

                # Calculate target and loss
                target = (noisy_latents - prev_latents)
                loss = F.mse_loss(target, noise)

                print(f"backpropagating loss: Step {global_step}: loss = {loss.item()}")

                # Simplify the backward pass - let accelerator handle mixed precision
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            
            # Update progress
            if accelerator.is_main_process:
                progress_bar.update(1)
                
                if global_step % 100 == 0:
                    print(f"Step {global_step}: loss = {loss.item()}")
                
                # Save checkpoint
                if global_step % 500 == 0:
                    # Unwrap the model
                    unwrapped_transformer = accelerator.unwrap_model(transformer)
                    
                    # Save LoRA weights
                    unwrapped_transformer.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    # display_image(pipeline)
            
            global_step += 1
            if global_step >= args.max_train_steps:
                break
    
    # Save final model
    if accelerator.is_main_process:
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        
        # Save LoRA weights
        unwrapped_transformer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")
    
    # Close wandb

def forward_for_training(
    self,
    hidden_states,
    timestep,
    pooled_projections,
    encoder_hidden_states,
    guidance,
    txt_ids,
    img_ids,
    sigmas = None,
    mu = None,
    num_inference_steps = None,
    num_warmup_steps = None,
    _num_timesteps = None,
):
    pass
def get_random_timestep(noise_scheduler):
    """
    Sample a random timestep for diffusion model training.
    
    During diffusion model training, each example only processes one random timestep
    per training iteration (not the full sequence). This function samples that timestep
    based on the noise scheduler's configuration.
    """
    return np.randint(
        0, noise_scheduler.config.num_train_timesteps
    )


def prepare_timesteps(image_seq_len, scheduler, device, num_inference_steps=1000, sigmas=None, mu=None):
    # 5. Prepare timesteps
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas

    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )

    num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)
    return timesteps, num_inference_steps, num_warmup_steps

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
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(cur_image[0].permute(1, 2, 0).cpu().numpy())
    ax1.set_title('Current Timestep')
    ax1.axis('off')
    
    ax2.imshow(prev_image[0].permute(1, 2, 0).cpu().numpy())
    ax2.set_title('Previous Timestep')
    ax2.axis('off')

    # Display noise - normalize to [0,1] range for visualization
    noise_viz = noise[0].permute(1, 2, 0).cpu().numpy()
    noise_viz = (noise_viz - noise_viz.min()) / (noise_viz.max() - noise_viz.min())
    ax3.imshow(noise_viz)
    ax3.set_title('Noise')
    ax3.axis('off')
    
    plt.show()

    


def get_noised_latent_at_random_timestep(sample_latents, scheduler, generator=None):
    """
    Get a noised latent at a random timestep along with its previous timestep value.
    """
    # Sample a random timestep
    num_train_timesteps = scheduler.config.num_train_timesteps
    timestep = torch.randint(
        0, num_train_timesteps, (sample_latents.shape[0],), 
        device=sample_latents.device, generator=generator
    ).long()
    timestep = torch.clamp(timestep, min=1)
    
    # Generate random noise and ensure it requires gradients
    noise = torch.randn_like(sample_latents, requires_grad=True)
    
    print("timestep: ", timestep)
    
    # Get noised latents at both timesteps
    cur_latents = scheduler.scale_noise(
        sample=sample_latents,
        timestep=timestep,
        noise=noise
    )
    
    # Calculate what the previous timestep would be
    prev_timestep = torch.clamp(timestep - 1, min=0)
    prev_latents = scheduler.scale_noise(
        sample=sample_latents,
        timestep=prev_timestep,
        noise=noise
    )        
    
    # Ensure all tensors have gradients enabled
    cur_latents.requires_grad_(True)
    prev_latents.requires_grad_(True)
    
    return cur_latents, prev_latents, noise, timestep


def display_image(pipeline):
    prompt = "Paul Mescal holding a sign that says hello world"
        
    try:
        with torch.inference_mode():
            image = pipeline(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=5,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
            
            # Save image with timestamp
            import time
            timestamp = int(time.time())
            save_path = f"generated_image_{timestamp}.png"
            image.save(save_path)
            print(f"Saved generated image to {save_path}")
            
    except Exception as e:
        print("Error during image generation:")
        print(e)
        print("Failed to generate image")


if __name__ == "__main__":
    train()