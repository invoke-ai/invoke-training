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
    prepare_model_for_kbit_training,
)
from diffusers.optimization import get_constant_schedule_with_warmup
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
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam optimizer")
    
    # parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision training")
    # LR scheduler
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup", choices=["cosine_with_restarts", "polynomial", "constant_with_warmup"], help="LR scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=0.1, help="LR warmup steps")

    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank for LoRA adaptation")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clip_grad_norm", type=float, default=0.0, help="Clip gradient norm")
    
    # Timestep sampler
    parser.add_argument("--timestep_sampler", type=str, default="shift", choices=["shift", "logit-normal", "mode-sampling", "cosma"], help="Timestep sampler")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Sigmoid scale for timestep sampler")
    parser.add_argument("--discrete_flow_shift", type=float, default=3.0, help="Discrete flow shift for timestep sampler")
    
    # Guidance scale
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for diffusion model")

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
    args.train_batch_size = 1  
    args.learning_rate = 1e-5 # Slightly lower learning rate for stability
    args.max_train_steps = 2000  # More steps for better convergence
    args.gradient_accumulation_steps = 8  # For effective batch size of 4
    args.use_8bit_adam = True  # Save memory with 8-bit optimizer
    args.mixed_precision = "no"  # Changed back to fp16
    args.clip_grad_norm = 0.0
    
    # Timestep sampler
    args.timestep_sampler = "shift"
    args.sigmoid_scale = 1.0
    args.discrete_flow_shift = 3.0

    # LR scheduler
    args.lr_scheduler = "constant_with_warmup"
    args.lr_warmup_steps = 0.1

    # Guidance scale
    args.guidance_scale = 1.0

    # LoRA parameters
    args.lora_rank = 4  # Higher rank for better adaptation capacity
    args.lora_alpha = 16  # Alpha = 2 * rank is a good rule of thumb
    args.lora_dropout = 0.05  # Standard dropout value for LoRA
    
    # Miscellaneous
    args.seed = 100  # Standard random seed
    
    return args
    
def train():
    args = parse_args(default=True)
    print(args)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,  # Always use fp16
        split_batches=True,
    )
    
    # Load pipeline with mixed precision
    print("Loading FLUX pipeline...")
    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name,
        torch_dtype=torch.float16,  # Always use float16
    )

    # Extract components
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
        # "to_out.0",  # Output projection
        # "ff.net.0.proj",  # MLP first projection
        # "ff.net.2",  # MLP second projection
    ]
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        inference_mode=False,
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
    
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps
    )


    # Prepare for accelerator
    print("PREPARE TRANSFORMER")
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    
    # Move models to device and ensure float16
    device = accelerator.device
    vae = vae.to(device).to(torch.float16)
    text_encoder = text_encoder.to(device).to(torch.float16)
    text_encoder_2 = text_encoder_2.to(device).to(torch.float16)
    transformer = transformer.to(device).to(torch.float16)
    
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
    # torch.manual_seed(55)
    transformer.train()
    while global_step < args.max_train_steps:
        print(f"global step: {global_step}")
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                # Process inputs in float16
                images = batch["images"].to(device, dtype=torch.float16)
                clip_input_ids = batch["clip_input_ids"].to(device)
                clip_attention_mask = batch["clip_attention_mask"].to(device)
                t5_input_ids = batch["t5_input_ids"].to(device)
                t5_attention_mask = batch["t5_attention_mask"].to(device)
                
                with torch.no_grad():
                    # Text encodings
                    clip_outputs = text_encoder(
                        clip_input_ids,
                        attention_mask=clip_attention_mask,
                        output_hidden_states=False
                    )
                    pooled_prompt_embeds = clip_outputs.pooler_output.to(torch.float16)
                    
                    t5_outputs = text_encoder_2(
                        t5_input_ids,
                        attention_mask=t5_attention_mask,
                        output_hidden_states=False
                    )[0]
                    prompt_embeds = t5_outputs.to(torch.float16)
                    
                    # Process latents
                    latents = vae.encode(images).latent_dist.sample().to(torch.float16) * 0.18215
                    
                    # Pack latents
                    batch_size, num_channels, height, width = latents.shape
                    latents = pipeline._pack_latents(latents, batch_size, num_channels, height, width)
                    guidance = torch.full((batch_size,), float(args.guidance_scale), device=accelerator.device)

                    img_ids = pipeline._prepare_latent_image_ids(
                        batch_size, height // 2, width // 2, device, torch.float16
                    )
                    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=torch.float16)   

                    if torch.any(torch.isnan(latents)):
                        print("latents: NAN")

                    # Get noised latents
                    # Sample noise that we'll add to the latents
                    noisy_latents, noise, timesteps, _ = get_noisy_model_input_and_timestep(
                        args,
                        latents=latents,
                        noise_scheduler=noise_scheduler,
                        device=device,
                        dtype=torch.float16
                    )

                    # Ensure float16
                    # latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    noisy_latents = noisy_latents.to(torch.float16)
                    noise = noise.to(torch.float16)

                # Forward pass
                # Forward pass through transformer
                def call_transformer(noisy_latents, timesteps, pooled_prompt_embeds, prompt_embeds, guidance, text_ids, img_ids):
                    with torch.set_grad_enabled(True), accelerator.autocast():
                        model_pred = transformer(
                            hidden_states=noisy_latents,
                            timestep=timesteps /1000,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            guidance=guidance,
                            txt_ids=text_ids,
                            img_ids=img_ids,
                            return_dict=False,
                        )[0]
                    return model_pred
                optimizer.zero_grad()
                model_pred = call_transformer(noisy_latents, timesteps, pooled_prompt_embeds, prompt_embeds, guidance, text_ids, img_ids)
                # Calculate loss
                target = noise - latents
                
                # LOSS FUNCTION
                # loss = train_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                if torch.any(torch.isnan(model_pred)):
                    print("model_pred: NAN")
                if torch.any(torch.isnan(target)):
                    print("target: NAN")

                loss = torch.nn.functional.mse_loss(model_pred, target)

                print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.8f}")
                print(f"backpropagating loss: Step {global_step}: loss = {loss.item()}")

                # Backward pass
                accelerator.backward(loss)
                
                # Check gradients
                has_bad_grads = check_gradients(transformer, global_step)
                
                # Optional: Skip update if gradients are bad
                if has_bad_grads:
                    print("WARNING: Bad gradients detected, skipping update")
                    optimizer.zero_grad()
                    global_step += 1
                    continue
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.clip_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                
                # Print gradient norms for specific layers (optional)
                if global_step % 100 == 0:
                    for name, param in transformer.named_parameters():
                        if param.grad is not None and "lora" in name.lower():
                            grad_norm = param.grad.norm().item()
                            print(f"Gradient norm for {name}: {grad_norm:.6f}")
                
            
            # Update progress
            if accelerator.is_main_process:
                progress_bar.update(1)
                
                if global_step % 100 == 0:
                    print(f"Step {global_step}: loss = {loss.item()}")
                
                # Save checkpoint
                if global_step % 500 == 0 and global_step > 0:
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
        2, noise_scheduler.config.num_train_timesteps
    )
def _get_random_timestep(noise_scheduler, latents):
    b_size = latents.shape[0]
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep

    timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    timesteps = timesteps.long().to(device)

    return timesteps


def get_noise_pred_and_target(noise_scheduler, latents):

    timesteps = _get_random_timestep(noise_scheduler, latents)
    return 

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


def get_sigmas(noise_scheduler, timesteps, device, n_dim, dtype):
    """
    Compute the sigmas for the given timesteps.
    """
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    return sigma
    

def get_noisy_model_input_and_timestep(args, latents, noise_scheduler, device, dtype):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1 
        - Section 3.1: The purpose of these samplers is to adaptively allocate more training samples 
        (in terms of sampling and loss weighting) to the harder parts of the problem,
        - TODO Options:
            - "Logit-Normal Sampling": 
            - "Mode Sampling with Heavy Tails": 
            - "CosMa": 
    """
    batch_size = latents.shape[0]
    noise = torch.randn_like(latents)

    # Sample Uniformly from [0, 1]
    if args.timestep_sampler == "shift":
        shift = args.discrete_flow_shift
        sigmas = torch.randn(batch_size, device=device)
        sigmas = sigmas * args.sigmoid_scale  # larger scale for more uniform sampling
        sigmas = sigmas.sigmoid()
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * noise_scheduler.config.num_train_timesteps
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)

    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
    return noisy_model_input.to(dtype), noise.to(dtype), timesteps.to(dtype), sigmas.to(dtype)

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

def check_gradients(model, step):
    """Helper function to check gradient statistics"""
    grad_stats = {
        "max": float("-inf"),
        "min": float("inf"),
        "mean": 0,
        "num_zero": 0,
        "num_nan": 0,
        "num_inf": 0,
        "total_params": 0
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            grad_stats["total_params"] += grad.numel()
            
            # Check for NaN and Inf
            grad_stats["num_nan"] += torch.isnan(grad).sum().item()
            grad_stats["num_inf"] += torch.isinf(grad).sum().item()
            
            # Skip statistics for NaN/Inf values
            valid_grad = grad[~torch.isnan(grad) & ~torch.isinf(grad)]
            if len(valid_grad) > 0:
                grad_stats["max"] = max(grad_stats["max"], valid_grad.max().item())
                grad_stats["min"] = min(grad_stats["min"], valid_grad.min().item())
                grad_stats["mean"] += valid_grad.mean().item()
                grad_stats["num_zero"] += (valid_grad == 0).sum().item()
    
    print(f"\nGradient Stats at step {step}:")
    print(f"Max grad: {grad_stats['max']:.6f}")
    print(f"Min grad: {grad_stats['min']:.6f}")
    print(f"Mean grad: {grad_stats['mean']:.6f}")
    print(f"Zero grads: {grad_stats['num_zero']} / {grad_stats['total_params']}")
    print(f"NaN grads: {grad_stats['num_nan']}")
    print(f"Inf grads: {grad_stats['num_inf']}\n")
    
    return grad_stats["num_nan"] > 0 or grad_stats["num_inf"] > 0

if __name__ == "__main__":
    train()