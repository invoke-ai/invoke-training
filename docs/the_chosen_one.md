# The Chosen One

Implementation of [The Chosen One](https://omriavrahami.com/the-chosen-one/).

Rough idea:

- Start with manual loop:
    - Generate images
    - Cluster the images, choose most cohesive cluster
    - Run LoRA training on the images
    - Generate images
    - Repeat

```bash

invoke-generate-images -o output/tco_1 -m /home/ryan/invokeai/autoimport/main/realisticVisionV51_v51VAE.safetensors --sd-version SD -p "A photo of a 50 years old man with curly hair" -n 128 --height 512 --width 512

python src/invoke_training/scripts/cluster_images.py -i output/tco_1 -o output/tco_cluster_1 -m /home/ryan/invokeai/models/any/clip_vision/ip_adapter_sd_image_encoder

invoke-dreambooth-lora-sd --cfg-file configs/tco_dreambooth_lora_sd_1x24gb.yaml

invoke-generate-images -o output/tco_2 -m /home/ryan/invokeai/autoimport/main/realisticVisionV51_v51VAE.safetensors -l "output/tco_training_1/1701374755.4400811/checkpoint_epoch-00000010.safetensors" --sd-version SD -p "A photo of a 50 years old man with curly hair" -n 128 --height 512 --width 512
```

```bash
invoke-generate-images -o output/tco_0_loop_0 -m /home/ryan/invokeai/autoimport/main/juggernautXL_version2.safetensors --sd-version SDXL -p "a cute mink with a brown jacket and red pants" -n 128 --height 1024 --width 1024

python src/invoke_training/scripts/cluster_images.py -i output/tco_0_loop_0 -o output/tco_0_loop_0_cluster -m /home/ryan/invokeai/models/any/clip_vision/ip_adapter_sd_image_encoder

# invoke-dreambooth-lora-sd --cfg-file configs/tco_dreambooth_lora_sd_1x24gb.yaml

# invoke-generate-images -o output/tco_2 -m /home/ryan/invokeai/autoimport/main/realisticVisionV51_v51VAE.safetensors -l "output/tco_training_1/1701374755.4400811/checkpoint_epoch-00000010.safetensors" --sd-version SD -p "A photo of a 50 years old man with curly hair" -n 128 --height 512 --width 512
```

## TODO

- Cluster images with DINOv2 instead of CLIP.
- Automatic image selection based on cluster cohesion focuses on a lot of the wrong things (e.g. how zoomed-in the charcater is). Experiment with manual selection schemes.
- Generate images for a variety of prompts to get more variety in the training images.
- Pivotal Tuning (or simultaneous TI and LoRA training)
