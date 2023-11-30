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
```