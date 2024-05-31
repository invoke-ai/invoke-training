import logging
import time

import torch

from invoke_training._shared.stable_diffusion.model_loading_utils import PipelineVersionEnum, load_pipeline
from invoke_training.model_merge.merge_models import WeightedSumMerger


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    base_model_path_1 = "/home/ryan/invokeai/autoimport/main/juggernautXL_version2.safetensors"
    pipeline_1 = load_pipeline(
        logger=logger,
        model_name_or_path=base_model_path_1,
        pipeline_version=PipelineVersionEnum.SDXL,
        torch_dtype=torch.float16,
        # variant=base_model_variant,
    )
    # base_model_path_2 = "stabilityai/stable-diffusion-xl-base-1.0"
    base_model_path_2 = "/home/ryan/src/invoke-training/lora_merge_output"
    pipeline_2 = load_pipeline(
        logger=logger,
        model_name_or_path=base_model_path_2,
        pipeline_version=PipelineVersionEnum.SDXL,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    submodel_names = ["unet", "text_encoder", "text_encoder_2"]

    pipelines = [pipeline_1, pipeline_2]
    for submodel_name in submodel_names:
        submodels: list[torch.nn.Module] = [getattr(pipeline, submodel_name) for pipeline in pipelines]
        submodel_state_dicts = [submodel.state_dict() for submodel in submodels]

        merger = WeightedSumMerger()
        merged_state_dict = merger.merge(submodel_state_dicts, [0.5, 0.5])

        # Merge the merged_state_dict back into the first pipeline to keep memory utilization low.
        submodel_state_dicts[0].update(merged_state_dict)
        submodels[0].load_state_dict(submodel_state_dicts[0], assign=True)
        logger.info(f"Merged {submodel_name} state_dicts.")

    # Run the merged pipeline.
    pipeline_1.to("cuda")
    image = pipeline_1(
        "A photo of a stuffed gnome at the beach with a pina colada.",
        num_inference_steps=30,
        generator=torch.Generator().manual_seed(0),
        height=1024,
        width=1024,
        # negative_prompt="",
    ).images[0]
    out_image_path = f"out_image_{time.time()}.png"
    image.save(out_image_path)
    logger.info(f"Saved image to '{out_image_path}'.")


if __name__ == "__main__":
    main()
