from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from censor_engine.libs.configs import get_config_path
from censor_engine.libs.detectors.box_based_detectors.multi_detectors import (
    NudeNetDetector,
)

from .dev_settings import DevConfig
from .file_settings import FileConfig
from .image_settings import AIConfig, RenderingConfig, ReverseCensorConfig
from .part_settings import (
    MergingConfig,
    PartInformationConfig,
    PartSettingsConfig,
)
from .video_settings import VideoConfig


@dataclass(slots=True)
class Config:
    # File Information
    dev_settings: DevConfig
    file_settings: FileConfig

    # Program Settings
    video_settings: VideoConfig
    rendering_settings: RenderingConfig
    ai_settings: AIConfig

    # Censor Information
    default_censor_settings: PartSettingsConfig
    censor_settings: PartInformationConfig
    reverse_censor: ReverseCensorConfig

    @staticmethod
    def _process_dict_data(config_data: dict) -> dict:
        # Common End Points
        dev_settings = config_data.get("dev_settings", {})
        file_settings = config_data.get("file_settings", {})
        video_settings = config_data.get("video_settings", {})
        render_settings = config_data.get("render_settings", {})
        ai_settings = config_data.get("ai_settings", {})
        censor_settings = config_data.get("censor_settings", {})

        # Censor Part Information
        enabled_parts = censor_settings.get("enabled_parts", [])
        default_part_settings = censor_settings.get(
            "default_part_settings", {}
        )
        reverse_censor_settings = censor_settings.get(
            "reverse_censor_settings", {}
        )
        merge_settings = censor_settings.get("merge_settings", {})

        # Shortcuts
        if isinstance(enabled_parts, str) and enabled_parts == "all":
            enabled_parts = list(
                NudeNetDetector.model_classifiers
            )  # HACK: make for more models
        elif isinstance(enabled_parts, str):
            enabled_parts = [enabled_parts]

        # # Handle Part Data
        parts_settings: dict[str, PartSettingsConfig] = {}
        default_settings_object = PartSettingsConfig(**default_part_settings)
        for part_name in enabled_parts:
            part_data = censor_settings.get(part_name, {})

            if (
                not part_data.get("censors")
                and part_data.get("state", "").lower() == "revealed"
            ):
                part_data["censors"] = []

            # Update Part Data
            if part_data:
                part_object = replace(default_settings_object, **part_data)
                part_object.__post_init__()
            else:
                part_object = default_settings_object
            parts_settings[part_name] = part_object

            # Part Corrects
            if parts_settings[part_name].name == "MISSING_NAME":
                parts_settings[part_name].name = part_name

        # Compile
        return {
            "dev_settings": DevConfig(**dev_settings),
            "file_settings": FileConfig(**file_settings),
            "video_settings": VideoConfig(**video_settings),
            "rendering_settings": RenderingConfig(**render_settings),
            "ai_settings": AIConfig(**ai_settings),
            "default_censor_settings": default_settings_object,
            "censor_settings": PartInformationConfig(
                enabled_parts=enabled_parts,
                parts_settings=parts_settings,
                merge_settings=MergingConfig(**merge_settings),
            ),
            "reverse_censor": ReverseCensorConfig(
                censors=reverse_censor_settings
            ),
        }

    @classmethod
    def from_yaml(cls, main_file_path, config_path: str) -> "Config":
        """
        Loads the YAML file and initializes the Config object

        Example:
            dev_settings:
                ...

            file_settings:
                ...

            ai_settings:
                ...

            image_settings:
                ...

            video_settings:
                ...

            render_settings:
                ...

            censor_settings:
                enabled_parts: [PART_A, PART_B, ...]

                merge_settings:
                    ...

                default_part_settings:
                    ...

                reverse_censor_settings:
                    ...

                PART_A:
                    ...

                PART_B:
                    ...


        """
        # Get the Full Config Path For Internal Configs
        full_config_path = get_config_path(config_path)

        # If it doesn't Exist, Check for Local Paths
        if not full_config_path.exists():
            full_config_path = Path(main_file_path) / config_path

        # If It still doesn't Exist, It must be Wrong
        if not full_config_path.exists():
            raise FileNotFoundError(
                f"Cannot find config at {config_path} or {full_config_path}"
            )

        # Load Config File
        with open(full_config_path) as file:
            config_data = yaml.safe_load(file) or {}

        return cls(**Config._process_dict_data(config_data))

    @classmethod
    def from_dictionary(cls, dict_config: dict[str, Any]) -> "Config":
        return cls(**Config._process_dict_data(dict_config))

    def _test_recalculate_missing_part_settings(self):
        existing_parts = self.censor_settings.parts_settings
        for part_name in self.censor_settings.enabled_parts:
            if not existing_parts.get(part_name):
                existing_parts[part_name] = self.default_censor_settings
