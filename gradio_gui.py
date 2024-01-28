# path/filename: gradio_project/config_interface.py

import gradio as gr
import subprocess
import yaml
import os
import threading
import time

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
    config = {
        "type": type,
        "seed": seed,
        "output": {
            "base_output_dir": base_output_dir
        },
        "optimizer": {
            "learning_rate": learning_rate,
            "optimizer_type": optimizer_type
        },
        "data_loader": {
            "type": data_loader_type,
            "dataset": {
                "type": dataset_type,
                "dataset_name": dataset_name
            },
            "image_resolution": image_resolution
        },
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
    }

    # Conditionally add optimizer-specific settings
    if optimizer_type == "Prodigy":
        config["optimizer"]["weight_decay"] = weight_decay
        config["optimizer"]["use_bias_correction"] = use_bias_correction
        config["optimizer"]["safeguard_warmup"] = safeguard_warmup
    elif optimizer_type == "AdamW":
        # For AdamW, add any specific settings or leave it as is
        pass

    # Conditionally update validation prompts based on whether multiple exist
    if validation_prompts:
        config["validation_prompts"] = validation_prompts.split(',')
    else:
        config["validation_prompts"] = []

    # Convert the config dictionary to a YAML string
    config_yaml = yaml.dump(config, sort_keys=False, default_flow_style=False)
    return config_yaml

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
                train_batch_size, num_validation_images_per_prompt, file_path):
    """
    Save the current configuration to a YAML file.
    """
    # Create the configuration using the create_config function
    config_yaml = create_config(type, seed, base_output_dir, learning_rate, optimizer_type, weight_decay, 
                                use_bias_correction, safeguard_warmup, data_loader_type, dataset_type, dataset_name, 
                                image_resolution, model, vae_model, train_text_encoder, cache_text_encoder_outputs, 
                                enable_cpu_offload_during_validation, gradient_accumulation_steps, mixed_precision, 
                                xformers, gradient_checkpointing, max_train_steps, save_every_n_epochs, 
                                save_every_n_steps, max_checkpoints, validation_prompts, validate_every_n_epochs, 
                                train_batch_size, num_validation_images_per_prompt)

    # Define the file path for saving the configuration
    if not file_path:
        file_path = "config.yaml"

    # Resolve the absolute path of the file
    full_file_path = os.path.abspath(file_path)

    # Write the YAML configuration to the file
    try:
        with open(full_file_path, "w") as file:
            file.write(config_yaml)
        return "Configuration saved successfully to " + full_file_path
    except Exception as e:
        return "Error saving configuration: " + str(e)

def run_training(config_yaml, output_textbox):
    command = [["invoke-train", "--cfg-file", config_yaml]]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    for line in iter(process.stdout.readline, ''):
        output_textbox.update(value=line)
    process.stdout.close()
    process.wait()

def invoke_training(type, seed, base_output_dir, learning_rate, optimizer_type, weight_decay, 
                    use_bias_correction, safeguard_warmup, data_loader_type, dataset_type, dataset_name, 
                    image_resolution, model, vae_model, train_text_encoder, cache_text_encoder_outputs, 
                    enable_cpu_offload_during_validation, gradient_accumulation_steps, mixed_precision, 
                    xformers, gradient_checkpointing, max_train_steps, save_every_n_epochs, 
                    save_every_n_steps, max_checkpoints, validation_prompts, validate_every_n_epochs, 
                    train_batch_size, num_validation_images_per_prompt, output_textbox):
    """
    Generate configuration data, save it to a YAML file, and invoke the training script.
    """
    # Generate the configuration using the create_config function
    config_yaml = create_config(type, seed, base_output_dir, learning_rate, optimizer_type, weight_decay, 
                                use_bias_correction, safeguard_warmup, data_loader_type, dataset_type, dataset_name, 
                                image_resolution, model, vae_model, train_text_encoder, cache_text_encoder_outputs, 
                                enable_cpu_offload_during_validation, gradient_accumulation_steps, mixed_precision, 
                                xformers, gradient_checkpointing, max_train_steps, save_every_n_epochs, 
                                save_every_n_steps, max_checkpoints, validation_prompts, validate_every_n_epochs, 
                                train_batch_size, num_validation_images_per_prompt)

    if not config_yaml or config_yaml.strip() == "":
        raise ValueError("Failed to generate configuration data.")

    config_file = "last_run_training_config.yaml"
    with open(config_file, "w") as file:
        file.write(config_yaml)

    # Start the training in a background thread
    thread = threading.Thread(target=run_training, args=(config_file, output_textbox), daemon=True)
    thread.start()


# Gradio Interface -- 
# for layout guidance, visit https://www.gradio.app/guides/controlling-layout
# for field/component docs, visit https://www.gradio.app/docs
with gr.Blocks() as demo:
    load_config_button = gr.File(label="Load Config", file_count="single")


    with gr.Row():
        # needs to be updated to conditionally control relevant fields based on type. currently lora sdxl
        type_input = gr.Dropdown(label="Type", choices=[
            "FINETUNE_LORA_SDXL", "FINETUNE_LORA_SD", "FINETUNE_LORA_AND_TI_SDXL", 
            "TEXTUAL_INVERSION_SD", "TEXTUAL_INVERSION_SDXL"
        ])
        seed_input = gr.Number(label="Seed", value=1)


    def change_optimizer_settings(optimizer_choice):
        if optimizer_choice == "Prodigy":
            return (
                gr.Number(label="Learning Rate", value=1.0),
                gr.Number(label="Weight Decay", value=0.01, visible=True),
                gr.Checkbox(label="Use Bias Correction", value=True, visible=True),
                gr.Checkbox(label="Safeguard Warmup", value=True, visible=True)
            )
        elif optimizer_choice == "AdamW":
            return (
                gr.Number(label="Learning Rate", value=4e-3),
                gr.Number(visible=False),  # Hide weight_decay_input
                gr.Checkbox(visible=False),  # Hide use_bias_correction_input
                gr.Checkbox(visible=False)   # Hide safeguard_warmup_input
            )
        else:
            return (None, None, None, None)


    with gr.Tab("Model Settings"):
        with gr.Row():
            model_input = gr.Textbox(label="Model", value="stabilityai/stable-diffusion-xl-base-1.0")
            vae_model_input = gr.Textbox(label="VAE Model", value="madebyollin/sdxl-vae-fp16-fix")
        with gr.Row():
            train_text_encoder_input = gr.Checkbox(label="Train Text Encoder", value=False)
            cache_text_encoder_outputs_input = gr.Checkbox(label="Cache Text Encoder Outputs", value=True)
            enable_cpu_offload_during_validation_input = gr.Checkbox(label="Enable CPU Offload During Validation", value=True)

    with gr.Tab("Optimizer"):
        with gr.Row():
            optimizer_type_input = gr.Dropdown(label="Optimizer Type", choices=["Prodigy", "AdamW"], value="Prodigy")
            learning_rate_input = gr.Number(label="Learning Rate", value=1)
        with gr.Row():
            weight_decay_input = gr.Number(label="Weight Decay")
            use_bias_correction_input = gr.Checkbox(label="Use Bias Correction")
            safeguard_warmup_input = gr.Checkbox(label="Safeguard Warmup")
        optimizer_type_input.change(
            fn=change_optimizer_settings, 
            inputs=optimizer_type_input, 
            outputs=[learning_rate_input, weight_decay_input, use_bias_correction_input, safeguard_warmup_input]
        )

    def update_dataset_input(dataset_type):
        if dataset_type == "Local Path":
            return gr.Textbox(label="Enter Dataset Local Path", value=default_local_path)
        else:
            return gr.Textbox(label="Dataset Name", value="Owner/RepoID (e.g., lambdalabs/pokemon-blip-captions)")

    def update_dataset_preview(dataset_type, dataset_name):
        if dataset_type == "Local Path":
            return gr.FileExplorer(label="Dataset Preview", root_dir=dataset_name, interactive=False, visible=True)
        else:
            return gr.FileExplorer(visible=False)

    default_local_path = os.path.join(os.getcwd(), "sample_data")

    with gr.Tab("Dataset Settings"):
        with gr.Column():
            data_loader_type_input = gr.Dropdown(label="Data Loader Type", choices=["IMAGE_CAPTION_SDXL_DATA_LOADER", "IMAGE_CAPTION_SD_DATA_LOADER"], value="IMAGE_CAPTION_SDXL_DATA_LOADER")
            image_resolution_input = gr.Number(label="Image Resolution", value=1024)
        with gr.Column():
            dataset_type_input = gr.Dropdown(label="Dataset Type", choices=["Hugging Face Dataset", "Local Path"], value="Local Path")
            dataset_name_input = gr.Textbox(label="Enter Dataset Local Path", value=default_local_path)
        with gr.Column():
            dataset_preview_panel = gr.FileExplorer(label="Dataset Preview", root_dir=default_local_path, interactive=False, height=300)
        dataset_type_input.change(fn=update_dataset_input, inputs=dataset_type_input, outputs=dataset_name_input)
        dataset_name_input.change(fn=update_dataset_preview, inputs=[dataset_type_input, dataset_name_input], outputs=dataset_preview_panel)

    with gr.Tab("Training Settings"):

        with gr.Row():

            with gr.Column():
                train_batch_size_input = gr.Number(label="Train Batch Size", value=1)
                gradient_accumulation_steps_input = gr.Number(label="Gradient Accumulation Steps", value=4)
                gradient_checkpointing_input = gr.Checkbox(label="Gradient Checkpointing", value=True)
            with gr.Column():
                mixed_precision_input = gr.Dropdown(label="Mixed Precision", choices=["fp16", "fp32"], value="fp16")
                xformers_input = gr.Checkbox(label="Xformers", value=True)

        with gr.Row():
            with gr.Column():
                max_train_steps_input = gr.Number(label="Training Steps", value=627)
                save_every_n_epochs_input = gr.Number(label="Save Every N Epochs", value=1)
                save_every_n_steps_input = gr.Number(label="Save Every N Steps", value=None)
                max_checkpoints_input = gr.Number(label="Max Checkpoints", value=100)

            with gr.Column():
                validate_every_n_epochs_input = gr.Number(label="Validate Every N Epochs", value=1)
                validation_prompts_input = gr.Textbox(label="Validation Prompts", placeholder="Enter prompts separated by commas")
                num_validation_images_per_prompt_input = gr.Number(label="Num Validation Images Per Prompt", value=3)

    generate_config_button = gr.Button("Preview Configuration")
    config_output = gr.Textbox(label="Generated Configuration", interactive=False)

    with gr.Tab("Run Training"):
        with gr.Row():
            base_output_dir_input = gr.Textbox(label="Base Output Dir", value="output/")
            run_button = gr.Button("Run Training Job")
        with gr.Row():
            training_output = gr.Label(label="Training Output")

        run_button.click(
            fn=invoke_training,
            inputs=[
                type_input, seed_input, base_output_dir_input, learning_rate_input, optimizer_type_input, weight_decay_input,
                use_bias_correction_input, safeguard_warmup_input, data_loader_type_input, dataset_type_input, dataset_name_input,
                image_resolution_input, model_input, vae_model_input, train_text_encoder_input, cache_text_encoder_outputs_input,
                enable_cpu_offload_during_validation_input, gradient_accumulation_steps_input, mixed_precision_input, xformers_input,
                gradient_checkpointing_input, max_train_steps_input, save_every_n_epochs_input, save_every_n_steps_input, 
                max_checkpoints_input, validation_prompts_input, validate_every_n_epochs_input, train_batch_size_input, 
                num_validation_images_per_prompt_input, training_output
            ],
            outputs=training_output
        )

    with gr.Tab("Save Config"):
        with gr.Row():
            file_path_input = gr.Textbox(label="Config File Path", value="config.yaml", placeholder="Enter the path to save the config")
            save_config_button = gr.Button("Save Config")
        with gr.Row():
            save_output = gr.Label()


    # Ensure that the configuration is generated before invoking the training script

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
            validate_every_n_epochs_input, train_batch_size_input, num_validation_images_per_prompt_input,
            file_path_input  # include the file path input
        ],
        outputs=save_output  # change to a Label for displaying the message
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()