import re


def split_pos_neg_prompts(prompt: str) -> tuple[str, str]:
    """Split a prompt of the following form into a positive prompt and a negative prompt:
    'positive prompt[negative prompt]'
    """
    prompt = prompt.strip()

    num_open_brackets = prompt.count("[")
    num_close_brackets = prompt.count("]")
    # If there are square brackets, then we try to extract the negative prompt.
    if num_open_brackets > 0 or num_close_brackets > 0:
        if num_open_brackets != 1 or num_close_brackets != 1:
            raise ValueError(
                "Failed to split the prompt into a positive and negative prompt. Expected exactly one pair of square "
                f"brackets. Prompt: '{prompt}'."
            )

        # Pattern to match a positive prompt followed by a negative prompt in square brackets.
        pattern = r"(.+)\[(.+)\]$"
        match = re.match(pattern, prompt)
        if match is None:
            raise ValueError(
                "Failed to split the prompt into a positive and negative prompt. Ensure that your prompts have the "
                f"form: 'positive prompt[negative prompt]'. Prompt: '{prompt}'."
            )
        return match.group(1), match.group(2)
    # There are no square brackets, so we treat as a single positive prompt.
    return prompt, ""


def merge_pos_neg_prompts(positive_prompt: str, negative_prompt: str) -> str:
    """Merge a positive prompt and a negative prompt into a single prompt of the form:
    'positive prompt[negative prompt]'
    """
    if "[" in positive_prompt or "]" in positive_prompt:
        raise ValueError(f"Positive prompt must not contain square brackets. Prompt: '{positive_prompt}'.")
    if "[" in negative_prompt or "]" in negative_prompt:
        raise ValueError(f"Negative prompt must not contain square brackets. Prompt: '{negative_prompt}'.")

    if negative_prompt == "":
        return positive_prompt
    return f"{positive_prompt}[{negative_prompt}]"


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
        if negative_prompt == "":
            negative_prompt = None
        negative_prompts.append(negative_prompt)

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
