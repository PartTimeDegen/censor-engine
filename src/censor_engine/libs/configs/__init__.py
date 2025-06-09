from pathlib import Path


def get_config_path(filename: str) -> Path:
    base_dir = Path(__file__).parent
    config_path = base_dir / filename
    return config_path.resolve()
