from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from censor_engine.libs.configs import get_config_path
from censor_engine.libs.detectors.box_based_detectors.nude_net import (
    NudeNetDetector,
)

from .development import DevelopmentConfig
from .file import FileConfig
from .image import AIConfig, RenderingConfig, ReverseCensorConfig
from .part import (
    MergingConfig,
    PartInformationConfig,
    PartSettingsConfig,
)
from .video import VideoConfig


class Config(BaseModel):
    """
    TODO: Write this.

    :param _type_ BaseModel: _description_
    :raises FileNotFoundError: _description_
    :return _type_: _description_
    """

    # File Information
    dev_settings: DevelopmentConfig
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

        # Handle "all" shortcut
        if isinstance(enabled_parts, str) and enabled_parts == "all":
            enabled_parts = list(NudeNetDetector.model_classifiers)
        elif isinstance(enabled_parts, str):
            enabled_parts = [enabled_parts]

        # Handle Part Data
        parts_settings: dict[str, PartSettingsConfig] = {}
        default_settings_object = PartSettingsConfig(**default_part_settings)

        for part_name in enabled_parts:
            part_data = censor_settings.get(part_name, {})

            if (
                not part_data.get("censors")
                and part_data.get("state", "").lower() == "revealed"
            ):
                part_data["censors"] = []

            # Merge defaults with overrides and RE-VALIDATE
            merged = default_settings_object.model_dump()
            merged.update(part_data)

            part_object = PartSettingsConfig.model_validate(merged)

            # Fix missing name
            if part_object.name == "MISSING_NAME":
                part_object.name = part_name

            parts_settings[part_name] = part_object

        # Compile into unified dict
        return {
            "dev_settings": DevelopmentConfig(**dev_settings),
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
    def from_yaml(cls, main_file_path: Path, config_path: str) -> "Config":
        """
        TODO: Write this.

        :param Path main_file_path: _description_
        :param str config_path: _description_
        :raises FileNotFoundError: _description_
        :return Config: _description_
        """
        full_config_path = get_config_path(config_path)

        if not full_config_path.exists():
            full_config_path = Path(main_file_path) / config_path

        if not full_config_path.exists():
            msg = f"Cannot find config at {config_path} or {full_config_path}"
            raise FileNotFoundError(msg)

        with Path.open(full_config_path) as file:
            config_data = yaml.safe_load(file) or {}

        return cls(**cls._process_dict_data(config_data))

    @classmethod
    def from_dictionary(cls, dict_config: dict[str, Any]) -> "Config":
        """
        TODO: Write this.

        :param dict[str, Any] dict_config: _description_
        :return Config: _description_
        """
        return cls(**cls._process_dict_data(dict_config))

    def _test_recalculate_missing_part_settings(self) -> None:
        existing_parts = self.censor_settings.parts_settings
        for part_name in self.censor_settings.enabled_parts:
            if not existing_parts.get(part_name):
                existing_parts[part_name] = self.default_censor_settings
