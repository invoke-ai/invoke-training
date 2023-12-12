# Tests

Run all unit tests with:

```bash
pytest tests/
```

There are some test 'markers' defined in [pyproject.toml](https://github.com/invoke-ai/invoke-training/blob/main/pyproject.toml) that can be used to skip some tests. For example, the following command skips tests that require a GPU or require downloading model weights:

```bash
pytest tests/ -m "not cuda and not loads_model"
```
