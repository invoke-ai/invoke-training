import torch
import tqdm
from peft.peft_model import PeftModel

# All original base model weights in a PeftModel have this prefix and suffix.
PEFT_BASE_LAYER_PREFIX = "base_model.model."
PEFT_BASE_LAYER_SUFFIX = ".base_layer.weight"


def get_patched_base_weights_from_peft_model(peft_model: PeftModel) -> dict[str, torch.Tensor]:
    """Get a state_dict containing the base model weights *thath are patched* in the provided PeftModel. I.e. only
    return base model weights that have associated LoRa layers, but don't return the LoRA layers.
    """
    state_dict = peft_model.state_dict()
    out_state_dict: dict[str, torch.Tensor] = {}
    for weight_name in state_dict:
        # Weights that end with ".base_layer.weight" are the original weights for LoRA layers.
        if weight_name.endswith(PEFT_BASE_LAYER_SUFFIX):
            # Extract the base module name.
            module_name = weight_name[: -len(PEFT_BASE_LAYER_SUFFIX)]
            assert module_name.startswith(PEFT_BASE_LAYER_PREFIX)
            module_name = module_name[len(PEFT_BASE_LAYER_PREFIX) :]

            out_state_dict[module_name] = state_dict[weight_name]

    return out_state_dict


def get_state_dict_diff(
    state_dict_1: dict[str, torch.Tensor], state_dict_2: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Return the difference between two state_dicts: state_dict_1 - state_dict_2."""
    return {key: state_dict_1[key] - state_dict_2[key] for key in state_dict_1}


@torch.no_grad()
def extract_lora_from_diffs(
    diffs: dict[str, torch.Tensor], rank: int, clamp_quantile: float, out_dtype: torch.dtype
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    lora_weights = {}
    for lora_name, mat in tqdm.tqdm(list(diffs.items())):
        # Use full precision for the intermediate calculations.
        mat = mat.to(torch.float32)

        is_conv2d = False
        if len(mat.shape) == 4:  # Conv2D
            is_conv2d = True
            out_dim, in_dim, kernel_h, kernel_w = mat.shape
            # Reshape to (out_dim, in_dim * kernel_h * kernel_w).
            mat = mat.flatten(start_dim=1)
        elif len(mat.shape) == 2:  # Linear
            out_dim, in_dim = mat.shape
        else:
            raise ValueError(f"Unexpected weight shape: {mat.shape}")

        # LoRA rank cannot exceed the original dimensions.
        assert rank < in_dim
        assert rank < out_dim

        u: torch.Tensor
        s: torch.Tensor
        v_h: torch.Tensor
        u, s, v_h = torch.linalg.svd(mat)

        # Apply the Eckart-Young-Mirsky theorem.
        # https://en.wikipedia.org/wiki/Low-rank_approximation#Proof_of_Eckart%E2%80%93Young%E2%80%93Mirsky_theorem_(for_Frobenius_norm)
        u = u[:, :rank]
        s = s[:rank]
        u = u @ torch.diag(s)

        v_h = v_h[:rank, :]

        # At this point, u is the lora_up (a.k.a. lora_B) weight, and v_h is the lora_down (a.k.a. lora_A) weight.
        # The reason we don't use more appropriate variable names is to keep memory usage low - we want the old tensors
        # to get cleaned up after each operation.

        # Clamp the outliers.
        dist = torch.cat([u.flatten(), v_h.flatten()])
        hi_val = torch.quantile(dist, clamp_quantile)
        low_val = -hi_val

        u = u.clamp(low_val, hi_val)
        v_h = v_h.clamp(low_val, hi_val)

        if is_conv2d:
            u = u.reshape(out_dim, rank, 1, 1)
            v_h = v_h.reshape(rank, in_dim, kernel_h, kernel_w)

        u = u.to(dtype=out_dtype).contiguous()
        v_h = v_h.to(dtype=out_dtype).contiguous()

        lora_weights[lora_name] = (u, v_h)
    return lora_weights
