import torch

from invoke_training.lora.injection.utils import find_modules, inject_lora_layers
from invoke_training.lora.layers import LoRALinearLayer
from invoke_training.lora.lora_block import LoRABlock


def test_find_modules_simple():
    """Test find_modules(...) behaviour on a simple ModuleDict structure."""
    # Construct mock module.
    linear1 = torch.nn.Linear(4, 8)
    linear2 = torch.nn.Linear(8, 16)
    conv1 = torch.nn.Conv2d(16, 32, 5)
    module = torch.nn.ModuleDict(
        {
            "linear1": linear1,
            "linear2": linear2,
            "conv1": conv1,
        }
    )

    result = list(find_modules(module, {torch.nn.Linear}))

    # Validate result.
    assert len(result) == 2
    result_by_name = {n: (n, p, m) for (n, p, m) in result}

    assert result_by_name["linear1"][0] == "linear1"
    assert result_by_name["linear1"][1] == module
    assert result_by_name["linear1"][2] == linear1

    assert result_by_name["linear2"][0] == "linear2"
    assert result_by_name["linear2"][1] == module
    assert result_by_name["linear2"][2] == linear2


def test_find_modules_nested():
    """Test find_modules(...) behaviour when target modules are nested."""
    # Construct mock module.
    linear1 = torch.nn.Linear(4, 8)
    linear2 = torch.nn.Linear(8, 16)
    conv1 = torch.nn.Conv2d(16, 32, 5)
    dict1 = torch.nn.ModuleDict({"linear2": linear2})
    module = torch.nn.ModuleDict(
        {
            "linear1": linear1,
            "dict1": dict1,
            "conv1": conv1,
        }
    )

    result = list(find_modules(module, {torch.nn.Linear, torch.nn.ModuleDict}))

    # Validate result.
    assert len(result) == 4
    result_by_name = {n: (n, p, m) for (n, p, m) in result}

    assert result_by_name[""][0] == ""
    assert result_by_name[""][1] is None
    assert result_by_name[""][2] == module

    assert result_by_name["dict1"][0] == "dict1"
    assert result_by_name["dict1"][1] == module
    assert result_by_name["dict1"][2] == dict1

    assert result_by_name["linear1"][0] == "linear1"
    assert result_by_name["linear1"][1] == module
    assert result_by_name["linear1"][2] == linear1

    assert result_by_name["dict1.linear2"][0] == "dict1.linear2"
    assert result_by_name["dict1.linear2"][1] == dict1
    assert result_by_name["dict1.linear2"][2] == linear2


def test_find_modules_include_descendants_of():
    """Test include_descendants_of parameter to find_modules(...)."""
    # Construct mock module.
    linear1 = torch.nn.Linear(4, 8)
    linear2 = torch.nn.Linear(8, 16)
    conv1 = torch.nn.Conv2d(16, 32, 5)
    list1 = torch.nn.ModuleList([conv1, linear2])
    module = torch.nn.ModuleDict(
        {
            "linear1": linear1,
            "list1": list1,
        }
    )

    # linear1 should be ignored, because it is not a descendant of a ModuleList.
    result = list(find_modules(module, {torch.nn.Linear}, include_descendants_of={torch.nn.ModuleList}))

    # Validate result.
    assert len(result) == 1
    result_by_name = {n: (n, p, m) for (n, p, m) in result}

    assert result_by_name["list1.1"][0] == "list1.1"
    assert result_by_name["list1.1"][1] == list1
    assert result_by_name["list1.1"][2] == linear2


def test_find_modules_exclude_descendants_of():
    """Test exclude_descendants_of parameter to find_modules(...)."""
    # Construct mock module.
    linear1 = torch.nn.Linear(4, 8)
    linear2 = torch.nn.Linear(8, 16)
    conv1 = torch.nn.Conv2d(16, 32, 5)
    list1 = torch.nn.ModuleList(
        [conv1, linear2]
    )  # linear2 should be ignored, because it is a descendant of a ModuleList.
    module = torch.nn.ModuleDict(
        {
            "linear1": linear1,
            "list1": list1,
        }
    )

    result = list(find_modules(module, {torch.nn.Linear}, exclude_descendants_of={torch.nn.ModuleList}))

    # Validate result.
    assert len(result) == 1
    result_by_name = {n: (n, p, m) for (n, p, m) in result}

    assert result_by_name["linear1"][0] == "linear1"
    assert result_by_name["linear1"][1] == module
    assert result_by_name["linear1"][2] == linear1


def test_find_modules_exclude_precedence_over_include():
    """Test that exclude_descendants_of takes precedence over include_descendants_of find_modules(...)."""
    # Construct mock module.
    linear1 = torch.nn.Linear(4, 8)
    list1 = torch.nn.ModuleList([linear1])
    module = torch.nn.ModuleDict({"list1": list1})

    # Test that when a descendant is excluded, exclude_descendants_of takes precedence over
    # include_descendants_of.
    result = list(
        find_modules(
            module,
            {torch.nn.Linear},
            include_descendants_of={torch.nn.ModuleDict},
            exclude_descendants_of={torch.nn.ModuleList},
        )
    )
    assert len(result) == 0

    # Test that when an ancestor is excluded, exclude_descendants_of takes precedence over
    # include_descendants_of.
    result = list(
        find_modules(
            module,
            {torch.nn.Linear},
            include_descendants_of={torch.nn.ModuleList},
            exclude_descendants_of={torch.nn.ModuleDict},
        )
    )
    assert len(result) == 0


def test_find_modules_duplicate():
    """Test that duplicate modules are only returned once."""
    # Construct mock module. Include linear1 twice.
    linear1 = torch.nn.Linear(4, 8)
    list1 = torch.nn.ModuleList([linear1])
    module = torch.nn.ModuleDict(
        {
            "linear1": linear1,
            "list1": list1,
        }
    )

    result = list(find_modules(module, {torch.nn.Linear}, exclude_descendants_of={torch.nn.ModuleList}))

    # Validate result.
    assert len(result) == 1
    result_by_name = {n: (n, p, m) for (n, p, m) in result}

    assert result_by_name["linear1"][0] == "linear1"
    assert result_by_name["linear1"][1] == module
    assert result_by_name["linear1"][2] == linear1


def test_inject_lora_layers():
    # Construct mock module.
    linear1 = torch.nn.Linear(4, 8)
    conv1 = torch.nn.Conv2d(16, 32, 5)
    module = torch.nn.ModuleDict(
        {
            "linear1": linear1,
            "conv1": conv1,
        }
    )

    lora_layers = inject_lora_layers(module, {torch.nn.Linear: LoRALinearLayer})

    assert len(lora_layers) == 1
    assert isinstance(module["linear1"], LoRABlock)
    assert module["linear1"].original_module == linear1
    assert module["linear1"].lora_layer._down.in_features == linear1.in_features
    assert module["linear1"].lora_layer._up.out_features == linear1.out_features
