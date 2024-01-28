# path/filename: gradio_project/config_interface.py

import gradio as gr
import subprocess
import yaml

def create_config(type, seed, base_output_dir, learning_rate, optimizer_type, weight_decay, 
                  use_bias_correction, safeguard_warmup, data_loader_type, dataset_type, dataset_name, 
                  image_resolution, model, vae_model, train_text_encoder, cache_text_encoder_outputs, 
                  enable_cpu_offload_during_validation, gradient_accumulation_steps, mixed_precision, 
                  xformers, gradient_checkpointing, max_train_steps, save_every_n_epochs, 
                  save_every_n_steps, max_checkpoints, validation_prompts, validate_every_n_epochs, 
                  train_batch_size, num_validation_images_per_prompt):
    """
    Create and return the configuration as a YAML-formatted string based on the inputs.
    """
def load_config(file_path):
    config_values = ["", 0, "", 0, "", 0, False, False, "", "", "", 0, "", "", False, False, False, 0, "", False, False, 0, 0, 0, 0, "", 0, 0, 0]

    if not file_path:
        return config_values

    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        return config_values

    if not isinstance(config, dict):
        return config_values

    config_values[0] = config.get("type", config_values[0])
    config_values[1] = config.get("seed", config_values[1])
    config_values[2] = config.get("base_output_dir", config_values[2])
    optimizer_config = config.get("optimizer", {})
    config_values[3] = optimizer_config.get("learning_rate", config_values[3])
    config_values[4] = optimizer_config.get("optimizer_type", config_values[4])
    config_values[5] = optimizer_config.get("weight_decay", config_values[5])
    config_values[6] = optimizer_config.get("use_bias_correction", config_values[6])
    config_values[7] = optimizer_config.get("safeguard_warmup", config_values[7])
    data_loader_config = config.get("data_loader", {})
    config_values[8] = data_loader_config.get("type", config_values[8])
    dataset_config = data_loader_config.get("dataset", {})
    config_values[9] = dataset_config.get("type", config_values[9])
    config_values[10] = dataset_config.get("dataset_name", config_values[10])
    config_values[11] = data_loader_config.get("image_resolution", config_values[11])
    config_values[12] = config.get("model", config_values[12])
    config_values[13] = config.get("vae_model", config_values[13])
    config_values[14] = config.get("train_text_encoder", config_values[14])
    config_values[15] = config.get("cache_text_encoder_outputs", config_values[15])
    config_values[16] = config.get("enable_cpu_offload_during_validation", config_values[16])
    config_values[17] = config.get("gradient_accumulation_steps", config_values[17])
    config_values[18] = config.get("mixed_precision", config_values[18])
    config_values[19] = config.get("xformers", config_values[19])
    config_values[20] = config.get("gradient_checkpointing", config_values[20])
    config_values[21] = config.get("max_train_steps", config_values[21])
    config_values[22] = config.get("save_every_n_epochs", config_values[22])
    config_values[23] = config.get("save_every_n_steps", config_values[23])
    config_values[24] = config.get("max_checkpoints", config_values[24])
    config_values[25] = ", ".join(config.get("validation_prompts", []))
    config_values[26] = config.get("validate_every_n_epochs", config_values[26])
    config_values[27] = config.get("train_batch_size", config_values[27])
    config_values[28] = config.get("num_validation_images_per_prompt", config_values[28])

    return config_values

def save_config(type, seed, base_output_dir, learning_rate, optimizer_type, weight_decay, 
                use_bias_correction, safeguard_warmup, data_loader_type, dataset_type, dataset_name, 
                image_resolution, model, vae_model, train_text_encoder, cache_text_encoder_outputs, 
                enable_cpu_offload_during_validation, gradient_accumulation_steps, mixed_precision, 
                xformers, gradient_checkpointing, max_train_steps, save_every_n_epochs, 
                save_every_n_steps, max_checkpoints, validation_prompts, validate_every_n_epochs, 
                train_batch_size, num_validation_images_per_prompt):
    """
    Save the current configuration to a YAML file.
    """

    config = {
        "type": type,
        "seed": seed,
        "output": {
            "base_output_dir": base_output_dir
        },
        "optimizer": {
            "learning_rate": learning_rate,
            "optimizer": {
                "optimizer_type": optimizer_type,
                "weight_decay": weight_decay,
                "use_bias_correction": use_bias_correction,
                "safeguard_warmup": safeguard_warmup
            }
        },
        # Additional configurations and validations might be added here
        "data_loader": {
            "type": data_loader_type,
            "dataset": {
                "type": dataset_type,
                "dataset_name": dataset_name
            },
            "image_transforms": {
                "resolution": image_resolution
            }
        },
        # General configurations
        "model": model,
        "vae_model": vae_model,
        "train_text_encoder": train_text_encoder,
        "cache_text_encoder_outputs": cache_text_encoder_outputs,
        "enable_cpu_offload_during_validation": enable_cpu_offload_during_validation,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "mixed_precision": mixed_precision,
        "xformers": xformers,
        "gradient_checkpointing": gradient_checkpointing,
        "max_train_steps": max_train_steps,
        "save_every_n_epochs": save_every_n_epochs,
        "save_every_n_steps": save_every_n_steps,
        "max_checkpoints": max_checkpoints,
        "validation_prompts": validation_prompts.split(','),
        "validate_every_n_epochs": validate_every_n_epochs,
        "train_batch_size": train_batch_size,
        "num_validation_images_per_prompt": num_validation_images_per_prompt
        # More fields might be added here for general configurations
    }
    config_yaml = yaml.dump(config, sort_keys=False, default_flow_style=False)
    return config_yaml

def invoke_training(config_output):
    """
    Check if configuration data is present, save it to a YAML file, and invoke the training script.
    """
    if not config_output or config_output.strip() == "":
        raise ValueError("No configuration data to write to the YAML file.")

    config_file = "training_config.yaml"
    with open(config_file, "w") as file:
        file.write(config_output)

    subprocess.run(["invoke-train", "--cfg-file", config_file])

# Gradio Interface -- 
# for layout guidance, visit https://www.gradio.app/guides/controlling-layout
# for field/component docs, visit https://www.gradio.app/docs
with gr.Blocks() as demo:
    load_config_button = gr.File(label="Load Config", file_count="single")
    save_config_button = gr.Button("Save Config")

    with gr.Row():
        # needs to be updated to conditionally control relevant fields based on type. currently lora sdxl
        type_input = gr.Dropdown(label="Type", choices=[
            "FINETUNE_LORA_SDXL", "FINETUNE_LORA_SD", "FINETUNE_LORA_AND_TI_SDXL", 
            "TEXTUAL_INVERSION_SD", "TEXTUAL_INVERSION_SDXL"
        ])
        seed_input = gr.Number(label="Seed", value=1)
        base_output_dir_input = gr.Textbox(label="Base Output Dir", value="output/")

    with gr.Row():
        optimizer_type_input = gr.Dropdown(label="Optimizer Type", choices=["Prodigy", "AdamW"], value="Prodigy")

    if optimizer_type_input.value == "Prodigy":
        learning_rate_input = gr.Number(label="Learning Rate", value=1.0)
        weight_decay_input = gr.Number(label="Weight Decay", value=0.01)
        use_bias_correction_input = gr.Checkbox(label="Use Bias Correction", value=True)
        safeguard_warmup_input = gr.Checkbox(label="Safeguard Warmup", value=True)
    else:
        learning_rate_input = gr.Number(label="Learning Rate", value=4e-3)



    with gr.Row():
        data_loader_type_input = gr.Dropdown(label="Data Loader Type", choices=["IMAGE_CAPTION_SDXL_DATA_LOADER","IMAGE_CAPTION_SD_DATA_LOADER"], value="IMAGE_CAPTION_SDXL_DATA_LOADER")
        # probably update this dataset type to be a dropdown that lets user select directory (or paste path)
        dataset_type_input = gr.Textbox(label="Dataset Type", value="HF_HUB_IMAGE_CAPTION_DATASET")
        dataset_name_input = gr.Textbox(label="Dataset Name", value="lambdalabs/pokemon-blip-captions")
        image_resolution_input = gr.Number(label="Image Resolution", value=1024)

    with gr.Row():
        model_input = gr.Textbox(label="Model", value="stabilityai/stable-diffusion-xl-base-1.0")
        vae_model_input = gr.Textbox(label="VAE Model", value="madebyollin/sdxl-vae-fp16-fix")
        train_text_encoder_input = gr.Checkbox(label="Train Text Encoder", value=False)
        cache_text_encoder_outputs_input = gr.Checkbox(label="Cache Text Encoder Outputs", value=True)
        enable_cpu_offload_during_validation_input = gr.Checkbox(label="Enable CPU Offload During Validation", value=True)

    with gr.Row():
        gradient_accumulation_steps_input = gr.Number(label="Gradient Accumulation Steps", value=4)
        mixed_precision_input = gr.Dropdown(label="Mixed Precision", choices=["fp16", "fp32"], value="fp16")
        xformers_input = gr.Checkbox(label="Xformers", value=True)

    with gr.Row():
        gradient_checkpointing_input = gr.Checkbox(label="Gradient Checkpointing", value=True)
        max_train_steps_input = gr.Number(label="Max Train Steps", value=627)
        save_every_n_epochs_input = gr.Number(label="Save Every N Epochs", value=1)

    with gr.Row():
        save_every_n_steps_input = gr.Number(label="Save Every N Steps", value=None)
        max_checkpoints_input = gr.Number(label="Max Checkpoints", value=100)
        validation_prompts_input = gr.Textbox(label="Validation Prompts", placeholder="Enter prompts separated by commas")

    with gr.Row():
        validate_every_n_epochs_input = gr.Number(label="Validate Every N Epochs", value=1)
        train_batch_size_input = gr.Number(label="Train Batch Size", value=1)
        num_validation_images_per_prompt_input = gr.Number(label="Num Validation Images Per Prompt", value=3)

    config_output = gr.Textbox(label="Generated Configuration", interactive=False)
    run_button = gr.Button("Run Training Job")

    # Ensure that the configuration is generated before invoking the training script
    generate_config_button = gr.Button("Generate Configuration")
    generate_config_button.click(
        fn=create_config, 
        inputs=[
            type_input, seed_input, base_output_dir_input, learning_rate_input, optimizer_type_input, weight_decay_input,
            use_bias_correction_input, safeguard_warmup_input, data_loader_type_input, dataset_type_input, dataset_name_input,
            image_resolution_input, model_input, vae_model_input, train_text_encoder_input, cache_text_encoder_outputs_input,
            enable_cpu_offload_during_validation_input, gradient_accumulation_steps_input, mixed_precision_input, xformers_input,
            gradient_checkpointing_input, max_train_steps_input, save_every_n_epochs_input, save_every_n_steps_input, 
            max_checkpoints_input, validation_prompts_input, validate_every_n_epochs_input, train_batch_size_input, 
            num_validation_images_per_prompt_input
        ],
        outputs=config_output
    )

    run_button.click(
        fn=invoke_training,
        inputs=[config_output],
        outputs=[]
    )

    # Bind Load Config button to function
    load_config_button.change(
        fn=load_config,
        inputs=[load_config_button],
        outputs=[
            type_input, seed_input, base_output_dir_input, learning_rate_input, optimizer_type_input, 
            weight_decay_input, use_bias_correction_input, safeguard_warmup_input, data_loader_type_input, 
            dataset_type_input, dataset_name_input, image_resolution_input, model_input, vae_model_input, 
            train_text_encoder_input, cache_text_encoder_outputs_input, enable_cpu_offload_during_validation_input, 
            gradient_accumulation_steps_input, mixed_precision_input, xformers_input, 
            gradient_checkpointing_input, max_train_steps_input, save_every_n_epochs_input, 
            save_every_n_steps_input, max_checkpoints_input, validation_prompts_input, 
            validate_every_n_epochs_input, train_batch_size_input, num_validation_images_per_prompt_input
        ]
    )

    # Bind Save Config button to function
    # TODO: This needs to be updated to give user a better save UX (selection, or notification of a named filepath at least)
    save_config_button.click(
        fn=save_config,
        inputs=[
            type_input, seed_input, base_output_dir_input, learning_rate_input, optimizer_type_input, 
            weight_decay_input, use_bias_correction_input, safeguard_warmup_input, data_loader_type_input, 
            dataset_type_input, dataset_name_input, image_resolution_input, model_input, vae_model_input, 
            train_text_encoder_input, cache_text_encoder_outputs_input, enable_cpu_offload_during_validation_input, 
            gradient_accumulation_steps_input, mixed_precision_input, xformers_input, 
            gradient_checkpointing_input, max_train_steps_input, save_every_n_epochs_input, 
            save_every_n_steps_input, max_checkpoints_input, validation_prompts_input, 
            validate_every_n_epochs_input, train_batch_size_input, num_validation_images_per_prompt_input
        ],
        outputs=gr.Textbox(label="Configuration YAML", interactive=False)
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()