# Model Merging

`invoke-training` provides utility scripts for several common model merging workflows. This page contains a summary of the available tools.

## `extract_lora_from_model_diff.py`

Extract a LoRA model that represents the difference between two base models.

Note that the extracted LoRA model is a lossy representation of the difference between the models, so some degradation in quality is expected.

For usage docs, run:
```bash
python src/invoke_training/model_merge/scripts/extract_lora_from_model_diff.py -h
```

## `merge_lora_into_model.py`

Merge a LoRA model into a base model to produce a new base model.

For usage docs, run:
```bash
python src/invoke_training/model_merge/scripts/merge_lora_into_model.py -h
```

## `merge_models.py`

Merge 2 or more base models to produce a single base model (using either LERP or SLERP). This is a simple merge strategy that merges all model weights in the same way.

For usage docs, run:
```bash
python src/invoke_training/model_merge/scripts/merge_models.py -h
```

## `merge_task_models_to_base_model.py`

Merge 1 or more task-specific base models into a single starting base model (using either [TIES](https://arxiv.org/abs/2306.01708) or [DARE](https://arxiv.org/abs/2311.03099)). This merge strategy aims to preserve the task-specific behaviors of the task models while making only small changes to the original base model. This approach enables multiple task models to be merged without excessive interference between them.

If you want to merge a task-specific LoRA into a base model using this strategy, first use `merge_lora_into_model.py` to produce a task-specific base model, then merge that new base model using this strategy.

For usage docs, run:
```bash
python src/invoke_training/model_merge/scripts/merge_task_models_to_base_model.py -h
```
