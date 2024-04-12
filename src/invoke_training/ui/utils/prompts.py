def split_prompt(prompt: str) -> tuple[str, str]:
    """
    Extracts two strings from an input string:
    - The first string contains everything outside square brackets.
    - The second string contains everything inside the square brackets.
    - square brackets are also removed from the prompt
    - square brackets can be escaped and ignored with a backslash
    """
    positive_prompt = ""
    negative_prompt = ""
    brackets_depth = 0
    escaped = False

    for char in prompt or "":
        if char == "[" and not escaped:
            negative_prompt += " "
            brackets_depth += 1
        elif char == "]" and not escaped:
            brackets_depth -= 1
            char = " "
        elif brackets_depth > 0:
            negative_prompt += char
        else:
            positive_prompt += char
            brackets_depth = 0

        # keep track of the escape char but only if it isn't escaped already
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False

    return positive_prompt, negative_prompt
