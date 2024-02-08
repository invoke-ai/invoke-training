from pathlib import Path

from invoke_training._shared.utils.jsonl import load_jsonl, save_jsonl


def test_jsonl_roundtrip(tmp_path: Path):
    in_objs = [{"a": 1, "b": 2}, {"a": 1, "b": 2}]
    jsonl_path = tmp_path / "test.jsonl"

    save_jsonl(in_objs, jsonl_path)
    out_objs = load_jsonl(jsonl_path)

    assert in_objs == out_objs
