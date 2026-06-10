from pathlib import Path
from typing import Any

from censor_engine.models.config import Config


def load_config(
    base_folder: Path,
    config_data: str | dict[str, Any],
) -> Config:
    if isinstance(config_data, str):
        return Config.from_yaml(base_folder, config_data)

    if isinstance(config_data, dict):
        return Config.from_dictionary(config_data)

    msg = "config_data must be either `str` or `dict[str, Any]`"
    raise TypeError(msg)
