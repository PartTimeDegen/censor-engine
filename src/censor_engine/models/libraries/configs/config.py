"""
This is the model for the config, which is used for changing the settings.

Full Example:
```
version: X

development:
    debug_level: X

file_handling:
    files:
        prefix: X
        suffix: X
    folders:
        uncensored: X
        censored: X

groups:
    persistence:
        - X
        - Y
        - Z
    merging:
        - X
        - Y
        - Z

engine_settings:
    processing:
        batch_size: X # TODO
        detection_downscale: X # TODO

    extra_ai_model_features:
        censor_layers: X
        body_segmentation: X

        clothes_segmentation: X
        focused_roi: X

image:
    merging:
        method: X
        merge_range: X # TODO

    reverse_censor:
        - X
        - Y
        - Z

video:
    fps:
        censoring: X # TODO
        output: X # TODO
    stability:
        size_difference_percentage: X
    persistance:
        censor_hold: X

detection:
    enabled_parts:
        - X
        - Y
        - Z

    default_settings: # NOTE: This is what I mean with {part_settings}
        mask: X

        minimum_score: X
        state: X
        protected_mask: X
        fade: X # TODO

        use_global_area: X # TODO

        censors:
            - X
            - Y
            - Z
        tracking_margin: X
            height: X
            width: X
        margins: X
            height: X
            width: X

    parts:
        PART_NAME:
            {part_settings}
```

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

import __main__
from censor_engine import PROJECT_ROOT

from ._constants import CURRENT_VERSION
from .settings._detection import DetectionSettings
from .settings._development import DevelopmentSettings
from .settings._engine_settings import EngineSettings
from .settings._file_handling import FileHandingSettings
from .settings._groups import GroupSettings
from .settings._image import ImageSettings
from .settings._video import VideoSettings


class Config(BaseModel):
    # Meta Stuff
    development: DevelopmentSettings = Field(
        default_factory=DevelopmentSettings
    )
    file_handling: FileHandingSettings = Field(
        default_factory=FileHandingSettings
    )

    # Processing Stuff
    groups: GroupSettings = Field(default_factory=GroupSettings)
    engine_settings: EngineSettings = Field(default_factory=EngineSettings)

    # AI Settings
    image: ImageSettings = Field(default_factory=ImageSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)

    version: int = CURRENT_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> Config:
        def get_data(
            base_path: Path,
            config_path: str | Path,
        ) -> Config | None:
            # Get Full Path
            full_config_path = base_path / config_path

            # Warn if Config Path doesn't Exist
            if not full_config_path.exists():
                msg = (
                    f'Couldn\'t Find Config "{config_path}" from "{base_path}"'
                )
                print(msg)  # noqa: T201
                return None

            # Return Data
            with Path.open(full_config_path) as file:
                config_data = yaml.safe_load(file)
            return cls.from_dict(config_data)

        # Check Built-in Configs
        base_dir = Path(PROJECT_ROOT) / "libraries" / "configs"
        data = get_data(base_dir, config_path)
        if data is not None:
            return data

        # Check Main.py Directory
        main_path = Path(__main__.__file__).resolve().parent
        data = get_data(main_path, config_path)
        if data is not None:
            return data

        msg = "Cannot Find Config!"
        raise FileNotFoundError(msg)
