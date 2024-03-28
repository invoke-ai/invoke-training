import peft
import torch


def fix_dora_init(peft_model: peft.PeftModel):
    # HACK: In the current PEFT DoRA implementation (peft==0.10.0), if a weight matrix contains a column of all
    # zeros, this will result in a divide-by-zero causing NaNs in the output tensor. This can occur if the base
    # model has undergone a precision conversion (e.g. from float32 to float16). In practice, it has been
    # observed with the SDXL 1.0 text_encoder_2 model in fp16 precision.
    # To work around this issue, we initialize the lora_B matrix to a small value (eps) to avoid the
    # divide-by-zero error. (Recall tha the lora_B matrix is typically initialized to all zeroes.)
    initialized_lora_b = False
    for param_name, param in peft_model.named_parameters():
        if "lora_B" in param_name:
            torch.nn.init.constant_(param, torch.finfo(param.dtype).eps)
            initialized_lora_b = True
    # This assertion is just here to make sure that the "lora_B" name matching is working as expected.
    assert initialized_lora_b
