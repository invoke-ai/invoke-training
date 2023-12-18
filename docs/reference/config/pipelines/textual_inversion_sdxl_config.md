# `TextualInversionSDXLConfig`

Below is a sample yaml config file for Textual Inversion SDXL training ([raw file](https://github.com/invoke-ai/invoke-training/blob/main/configs/textual_inversion_sdxl_gnome_1x24gb_example.yaml)). All of the configuration fields are explained in detail on this page.

```yaml title="textual_inversion_sdxl_gnome_1x24gb_example.yaml"
--8<-- "configs/textual_inversion_sdxl_gnome_1x24gb_example.yaml"
```

<!-- To control the member order, we first list out the members whose order we care about, then we list the rest. -->
::: invoke_training.config.pipelines.textual_inversion_config.TextualInversionSDXLConfig
    options:
      members:
      - type
      - seed
      - output
      - optimizer
      - data_loader
      - model

::: invoke_training.config.pipelines.textual_inversion_config.TextualInversionSDXLConfig
    options:
      filters:
      - "!^type"
      - "!^seed"
      - "!^output"
      - "!^optimizer"
      - "!^data_loader"
      - "!^model"
