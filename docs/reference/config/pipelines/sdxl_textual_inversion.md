# `SdxlTextualInversionConfig`

Below is a sample yaml config file for Textual Inversion SDXL training ([raw file](https://github.com/invoke-ai/invoke-training/blob/main/configs/textual_inversion_sdxl_gnome_1x24gb_example.yaml)). All of the configuration fields are explained in detail on this page.

```yaml title="textual_inversion_sdxl_gnome_1x24gb_example.yaml"
--8<-- "configs/textual_inversion_sdxl_gnome_1x24gb_example.yaml"
```

<!-- To control the member order, we first list out the members whose order we care about, then we list the rest. -->
::: invoke_training.pipelines.stable_diffusion_xl.textual_inversion.config.SdxlTextualInversionConfig
    options:
      members:
      - type

<!-- Note that we always hide "model_config", as it should not be set by the user. -->
::: invoke_training.pipelines.stable_diffusion_xl.textual_inversion.config.SdxlTextualInversionConfig
    options:
      filters:
      - "!^model_config"
      - "!^type"
