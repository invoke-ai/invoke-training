import glob
from pathlib import Path


def get_pipeline_config_file_paths() -> list[str]:
    """A helper function that returns the paths of all the sample config files."""

    cur_file = Path(__file__)
    config_dir = cur_file.parent.parent.parent.parent.parent / "src/invoke_training/sample_configs"
    config_files = glob.glob(str(config_dir) + "/**/*.yaml", recursive=True)

    assert len(config_files) > 0

    return config_files
