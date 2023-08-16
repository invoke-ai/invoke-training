import typing


def flatten_config(val: typing.Any, parent_key: str = "", separator: str = ".") -> dict:
    """Flatten all lists and dicts in a nested config dict by chaining config keys/indexes using the `separator`.

    For example:
    ```
    >>> val = {"a": 1, "b": {"c": 2}, "d": [3, 4]}
    >>> flatten_dict(val)
    {'a': 1, 'b.c': 2, "d.0": 3, "d.1": 4}
    ```
    Args:
        val (any): The config to flatten (or value of a dict in recursive calls).
        parent_key (str, optional): The parent key to chain with keys in `val`.
        separator (str, optional): The separator to use for key chaining. Defaults to ".".

    Returns:
        dict: A flattened dict.
    """
    if isinstance(val, dict):
        child_key_value_pairs = val.items()
    elif isinstance(val, (list, tuple)):
        child_key_value_pairs = enumerate(val)
    else:  # val can't be flattened further.
        return {parent_key: val}

    flat_dict = {}
    for k, v in child_key_value_pairs:
        new_key = parent_key + separator + str(k) if parent_key else str(k)
        flat_dict.update(flatten_config(v, new_key, separator))
    return flat_dict
