import typing

import torch

from invoke_training.core.lora.injection.lora_layer_collection import LoRALayerCollection
from invoke_training.core.lora.layers import BaseLoRALayer
from invoke_training.core.lora.lora_block import LoRABlock


def find_modules(
    module: torch.nn.Module,
    targets: typing.Set[typing.Type[torch.nn.Module]],
    include_descendants_of: typing.Optional[typing.Set[typing.Type[torch.nn.Module]]] = None,
    exclude_descendants_of: typing.Optional[typing.Set[typing.Type[torch.nn.Module]]] = None,
    memo: typing.Optional[typing.Set[torch.nn.Module]] = None,
    prefix: str = "",
    parent: typing.Optional[torch.nn.Module] = None,
) -> typing.Iterator[typing.Tuple[str, torch.nn.Module, torch.nn.Module]]:
    """Find sub-modules of 'module' that satisfy the search criteria.
    Args:
        module (torch.nn.Module): The base module whose sub-modules will be searched.
        targets (typing.Set[typing.Type[torch.nn.Module]]): The set of module types to search for.
        include_descendants_of (typing.Set[typing.Type[torch.nn.Module]], optional): If set, then only
            descendants of these types (and their subclasses) will be searched. 'exclude_descendants_of' takes
            precedence over 'include_descendants_of'.
        exclude_descendants_of (typing.Set[typing.Type[torch.nn.Module]], optional): If set, then the
            descendants of these types (and their subclasses) will be ignored in the search. 'exclude_descendants_of'
            takes precedence over 'include_descendants_of'.
        memo (typing.Set[torch.nn.Module], optional): A memo to store the set of modules already visited in the search.
            memo is typically only set in recursive calls of this function.
        prefix (str, optional): A prefix that will be added to the module name.
        parent (torch.nn.Module, optional): The parent of 'module'. This is used for tracking the parent in recursive
            calls to this function so that it can be returned along with the module.
    Yields:
        typing.Tuple[str, torch.nn.Module, torch.nn.Module]: A tuple (name, parent, module) that matches the search
            criteria.
    """

    if memo is None:
        memo = set()

    if module in memo:
        # We've already visited this module in the search.
        return

    memo.add(module)

    # If we have hit an excluded module type, do not search any further.
    # Note that this takes precedence over include_descendants_of.
    if exclude_descendants_of is not None and isinstance(module, tuple(exclude_descendants_of)):
        return

    # If the include_descendants_of requirement is already satisfied, and this module matches a target class, then all
    # of the search criteria are satisfied, so yield it.
    if include_descendants_of is None and isinstance(module, tuple(targets)):
        yield prefix, parent, module

    # The include_descendants_of requirement is NOT YET satisfied. Check if this module satisfies it.
    updated_include_descendants_of = include_descendants_of
    if include_descendants_of is not None and isinstance(module, tuple(include_descendants_of)):
        # Drop the include_descendants_of requirement if this module satisfied it.
        updated_include_descendants_of = None

    # Recursively search the child modules.
    for child_name, child_module in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + child_name
        yield from find_modules(
            module=child_module,
            targets=targets,
            include_descendants_of=updated_include_descendants_of,
            exclude_descendants_of=exclude_descendants_of,
            memo=memo,
            prefix=submodule_prefix,
            parent=module,
        )


def inject_lora_layers(
    module: torch.nn.Module,
    lora_map: typing.Dict[type[torch.nn.Module], type[BaseLoRALayer]],
    include_descendants_of: typing.Optional[typing.Set[typing.Type[torch.nn.Module]]] = None,
    exclude_descendants_of: typing.Optional[typing.Set[typing.Type[torch.nn.Module]]] = None,
    prefix: str = "",
    dtype: torch.dtype = None,
    lora_rank_dim: int = 4,
) -> LoRALayerCollection:
    """Iterates over all of the modules in 'module' and if they are present in 'replace_map' then replaces them with the
    mapped LoRA layer type.
    Args:
        module (torch.nn.Module): The original module that will be monkeypatched.
        lora_map (typing.Dict[type[torch.nn.Module], type[torch.nn.Module]]): A mapping from module types that should
            have LoRA layers added to the type of LoRA layers that should be used.
            Example:
            ```
            lora_map = {torch.nn.Linear: LoRALinearLayer}
            ```
        include_descendants_of (typing.Set[typing.Type[torch.nn.Module]], optional): Forwarded to find_modules(...).
        exclude_descendants_of (typing.Set[typing.Type[torch.nn.Module]], optional): Forwarded to find_modules(...).
        prefix (str, optional): A prefix that will be added to the names of all of the LoRA layers.
        dtype (torch.dtype, optional): The dtype to construct the new layer with.
        lora_rank_dim (int, optional): The rank dimension to use for the injected LoRA layers.
    Returns:
        LoRALayerCollection: A ModuleDict of all of the LoRA layers that were injected into the module.
    """
    lora_layers = LoRALayerCollection()

    for name, parent, module in find_modules(
        module=module,
        targets=lora_map.keys(),
        include_descendants_of=include_descendants_of,
        exclude_descendants_of=exclude_descendants_of,
        prefix=prefix,
    ):
        # Lookup the LoRA class to use.
        lora_layer_cls = lora_map[type(module)]

        # Initialize the LoRA layer with the correct dimensions.
        lora_layer = lora_layer_cls.from_layer(module, rank=lora_rank_dim, dtype=dtype)

        # Join the LoRA layer and the original layer in a block.
        lora_block = LoRABlock(original_module=module, lora_layer=lora_layer)

        # Monkey-patch the parent module with the new LoRA block.
        child_field_name = name.split(".")[-1]
        setattr(
            parent,
            child_field_name,
            lora_block,
        )

        lora_layers.add_layer(lora_layer, name)

    return lora_layers
