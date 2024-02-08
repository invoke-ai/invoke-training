import json
from pathlib import Path
from typing import Any


def load_jsonl(jsonl_path: Path | str) -> list[Any]:
    """Load a JSONL file."""
    data = []
    with open(jsonl_path) as f:
        while (line := f.readline()) != "":
            data.append(json.loads(line))
    return data


def save_jsonl(data: list[Any], jsonl_path: Path | str) -> None:
    """Save a list of objects to a JSONL file."""
    with open(jsonl_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
