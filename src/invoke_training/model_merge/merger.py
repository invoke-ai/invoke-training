import torch

StateDict = dict[str, torch.Tensor]


def normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    return [weight / total for weight in weights]


class WeightedSumMerger:
    def merge(self, state_dicts: list[StateDict], weights: list[float]) -> StateDict:
        if len(state_dicts) < 2:
            raise ValueError("Must provide >=2 models to merge.")

        if len(state_dicts) != len(weights):
            raise ValueError("Must provide a weight for each model.")

        normalized_weights = normalize_weights(weights)

        out_state_dict: StateDict = {}

        for key in state_dicts[0].keys():
            tensors = [state_dict[key] for state_dict in state_dicts]
            out_state_dict[key] = self._merge_tensors(tensors, normalized_weights)

        return out_state_dict

    def _merge_tensors(self, tensors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
        return sum(weight * tensor for weight, tensor in zip(weights, tensors, strict=True))
