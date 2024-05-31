import gradio as gr
from src.invoke_training._shared.stable_diffusion.model_loading_utils import PipelineVersionEnum

class ModelMergePage:
    def __init__(self):
        with gr.Blocks(
            title="Model Merging",
            analytics_enabled=False,
            head='<link rel="icon" type="image/x-icon" href="/assets/favicon.png">',
        ) as app:
            with gr.Tab(label="Merge LoRA into SD Model"):
                self._create_merge_tab()
            with gr.Tab(label="Extract LoRA from Checkpoint"):
                self._create_extract_tab()

        self._app = app

    def _create_merge_tab(self):
        with gr.Row():
            gr.Markdown("## Merge LoRA into SD Model")
        with gr.Row():
            gr.Markdown("## Merge LoRA into SD Model")
        with gr.Row():
            base_model = gr.Textbox(label="Base Model Path")
            base_model_variant = gr.Textbox(label="Base Model Variant (Optional)")
            base_model_type = gr.Dropdown(choices=["SD", "SDXL"], label="Base Model Type")
            lora_models = gr.Textbox(label="LoRA Models (comma-separated paths with optional weights, e.g., 'path1::0.5,path2')")
            output_path = gr.Textbox(label="Output Path")
            save_dtype = gr.Dropdown(choices=["float32", "float16", "bfloat16"], label="Save Dtype")
            merge_button = gr.Button("Merge")

        merge_button.click(
            fn=self._merge_lora_into_sd_model,
            inputs=[base_model, base_model_variant, base_model_type, lora_models, output_path, save_dtype],
            outputs=[]
        )

    def _create_extract_tab(self):
        with gr.Row():
            gr.Markdown("## Extract LoRA from Checkpoint")
        with gr.Row():
            gr.Markdown("## Extract LoRA from Checkpoint")
        with gr.Row():
            model_type = gr.Dropdown(choices=["sd1", "sdxl"], label="Model Type")
            model_orig = gr.Textbox(label="Original Model Path")
            model_tuned = gr.Textbox(label="Tuned Model Path")
            save_to = gr.Textbox(label="Save To Path")
            load_precision = gr.Dropdown(choices=["fp32", "fp16", "bf16"], label="Load Precision")
            save_precision = gr.Dropdown(choices=["fp32", "fp16", "bf16"], label="Save Precision")
            device = gr.Dropdown(choices=["cuda", "cpu"], label="Device")
            lora_rank = gr.Slider(minimum=1, maximum=128, step=1, label="LoRA Rank")
            clamp_quantile = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Clamp Quantile")
            extract_button = gr.Button("Extract")

        extract_button.click(
            fn=self._extract_lora_from_checkpoint,
            inputs=[model_type, model_orig, model_tuned, save_to, load_precision, save_precision, device, lora_rank, clamp_quantile],
            outputs=[]
        )

    def _merge_lora_into_sd_model(self, base_model, base_model_variant, base_model_type, lora_models, output_path, save_dtype):
        lora_models_list = [tuple(lm.split("::")) if "::" in lm else (lm, 1.0) for lm in lora_models.split(",")]
        lora_models_list = [(path, float(weight)) for path, weight in lora_models_list]
        from src.invoke_training.scripts._experimental.lora_merge.merge_lora_into_sd_model import merge_lora_into_sd_model
        import logging

        logger = logging.getLogger(__name__)
        merge_lora_into_sd_model(
            logger=logger,
            base_model=base_model,
            base_model_variant=base_model_variant,
            base_model_type=PipelineVersionEnum(base_model_type),
            lora_models=lora_models_list,
            output=output_path,
            save_dtype=save_dtype,
        )

    def _extract_lora_from_checkpoint(self, model_type, model_orig, model_tuned, save_to, load_precision, save_precision, device, lora_rank, clamp_quantile):
        from src.invoke_training.scripts._experimental.lora_extraction.extract_lora_from_checkpoint import extract_lora
        import logging

        logger = logging.getLogger(__name__)
        extract_lora(
            logger=logger,
            model_type=model_type,
            model_orig_path=model_orig,
            model_tuned_path=model_tuned,
            save_to=save_to,
            load_precision=load_precision,
            save_precision=save_precision,
            device=device,
            lora_rank=lora_rank,
            clamp_quantile=clamp_quantile,
        )

    def app(self):
        return self._app
