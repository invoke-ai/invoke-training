def parse_model_arg(model: str, delimiter: str = "::") -> tuple[str, str | None]:
    """Parse a model argument into a model and a variant."""
    parts = model.split(delimiter)
    if len(parts) == 1:
        return parts[0], None
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        raise ValueError(f"Unexpected format for --models arg: '{model}'.")
