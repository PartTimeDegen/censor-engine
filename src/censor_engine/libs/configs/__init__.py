import os
import yaml


def get_config_path(filename: str) -> str:
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, filename)
    return os.path.abspath(config_path)
