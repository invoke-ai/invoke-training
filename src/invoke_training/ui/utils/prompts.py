NEGATIVE_PROMPT_DELIMITER = "[NEG]"


def split_pos_neg_prompts(prompt: str) -> tuple[str, str]:
    """Split a prompt containing a '[NEG]' delimiter into a positive prompt and a negative prompt.

    Examples:
    - 'positive prompt[NEG]negative prompt'     -> ('positive prompt', 'negative prompt')
    - 'positive prompt'                         -> ('positive prompt', '')
    - 'positive prompt[NEG]negative[NEG]prompt' -> Raises ValueError
    """
    prompt = prompt.strip()

    splits = prompt.split(NEGATIVE_PROMPT_DELIMITER)
    if len(splits) == 1:
        # This is a positive prompt only.
        return splits[0], ""
    elif len(splits) == 2:
        # This is a positive prompt followed by a negative prompt.
        return splits[0], splits[1]

    raise ValueError(
        f"Failed to split the prompt into a positive and negative prompt. Expected at most one "
        f"'{NEGATIVE_PROMPT_DELIMITER}' delimiter. Prompt: '{prompt}'."
    )


def merge_pos_neg_prompts(positive_prompt: str, negative_prompt: str) -> str:
    """Merge a positive prompt and a negative prompt into a single prompt of the form:
    'positive prompt[NEG]negative prompt'
    """
    if NEGATIVE_PROMPT_DELIMITER in positive_prompt:
        raise ValueError(
            f"Positive prompt cannot contain the '{NEGATIVE_PROMPT_DELIMITER}' delimiter. Prompt: '{positive_prompt}'"
        )
    if NEGATIVE_PROMPT_DELIMITER in negative_prompt:
        raise ValueError(
            f"Negative prompt cannot contain the '{NEGATIVE_PROMPT_DELIMITER}' delimiter. Prompt: '{negative_prompt}'"
        )

    if negative_prompt == "":
        return positive_prompt

    return f"{positive_prompt}{NEGATIVE_PROMPT_DELIMITER}{negative_prompt}"


def convert_ui_prompts_to_pos_neg_prompts(prompts: str) -> tuple[list[str], list[str | None]]:
    """Convert prompts from the UI textbox format to lists of positive and negative prompts."""

    ui_prompt_list = prompts.split("\n")
    positive_prompts = []
    negative_prompts = []
    for prompt in ui_prompt_list:
        positive_prompt, negative_prompt = split_pos_neg_prompts(prompt)

        # Skip empty lines.
        if positive_prompt == "" and negative_prompt == "":
            continue

        positive_prompts.append(positive_prompt)
        negative_prompts.append(negative_prompt)

    # Convert empty negative prompts to None.
    negative_prompts = [neg if neg != "" else None for neg in negative_prompts]
    # Convert negative_prompts to a single None if all negative prompts are None.
    if all([p is None for p in negative_prompts]):
        negative_prompts = None
    return positive_prompts, negative_prompts


def convert_pos_neg_prompts_to_ui_prompts(
    positive_prompts: list[str], negative_prompts: list[str | None] | None
) -> str:
    """Convert lists of positive and negative prompts to the UI textbox format."""
    # Convert `list[str | None] | None` to `list[str]`.
    if negative_prompts is None:
        negative_prompts = [""] * len(positive_prompts)
    negative_prompts: list[str] = [neg if neg is not None else "" for neg in negative_prompts]

    ui_prompts = ""
    for positive_prompt, negative_prompt in zip(positive_prompts, negative_prompts, strict=True):
        ui_prompts += merge_pos_neg_prompts(positive_prompt, negative_prompt) + "\n"
    return ui_prompts.strip()
