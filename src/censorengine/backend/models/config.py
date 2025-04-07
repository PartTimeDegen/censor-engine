from dataclasses import dataclass, field, replace
import os
from typing import Any, Optional

import yaml
from censorengine.backend.constants.files import CONFIGS_FOLDER
from censorengine.backend.models.structures.schemas import Censor
from censorengine.backend.models.structures.enums import PartState
from censorengine.backend.models.tools.debugger import DebugLevels


# Control Settings Stuff
@dataclass
class DevConfig:
    debug_level: DebugLevels = DebugLevels.NONE


@dataclass
class FileConfig:
    file_prefix: str = ""
    file_suffix: str = ""

    uncensored_folder: str = "uncensored"
    censored_folder: str = "censored"


@dataclass
class ImageConfig:
    smoothing: bool = True


@dataclass
class VideoConfig:
    # Core Settings
    # NOTE: The default "-1" means to use the native FPS
    censoring_fps: int = -1
    output_fps: int = -1

    # Video Cleaning Settings
    # # Frame Stability Config
    frame_difference_threshold: float = 0.05
    size_change_tolerance: float = 0.80

    # Frame Part Persistence Config
    part_frame_hold: int = 1


@dataclass
class RenderingConfig:
    smoothing: bool = True


@dataclass
class AIConfig:
    ai_model_downscale_factor: int = 1  # TODO: Implement


@dataclass
class MergingConfig:
    merge_range: float | int = -1.0
    merge_groups: list[list[str]] = field(default_factory=list)

    def __post_init__(self):
        # Type Narrow to Float
        if isinstance(self.merge_range, int):
            self.merge_range = float(self.merge_range)


# Parts Stuff
@dataclass
class PartSettingsConfig:
    # Meta
    name: str = "MISSING_NAME"

    # Settings
    minimum_score: Optional[float] = 0.20
    censors: list[Censor] = field(default_factory=list)
    shape: str = "box"
    margin: int | float | dict[str, float] = 0.0
    state: PartState = PartState.UNPROTECTED
    protected_shape: Optional[str] = None
    fade_percent: int = 0  # 0 - 100

    # Semi Meta Settings
    use_global_area: bool = True

    def __str__(self):
        return self.name

    def __post_init__(self):
        """
        This section is to handle that the incoming data are Python builtins,
        not for example Censor or PartState.

        TODO: This doesn't confirm information is the right type, use pydantic
        at some point.

        """
        # Censors
        self.censors = [
            Censor(**censor) if isinstance(censor, dict) else censor
            for censor in self.censors
        ]

        # Part State
        if isinstance(self.state, str):
            try:
                self.state = PartState[self.state.upper()]
            except ValueError:
                raise ValueError(f"Invalid PartState value: {self.state}")

        # Margin
        if not isinstance(self.margin, (int, float, dict)):
            raise TypeError(f"Invalid type for margin: {type(self.margin)}")


@dataclass
class ReverseCensorConfig:
    censors: list[Censor] = field(default_factory=list)

    def __post_init__(self):
        """
        Ensure censors are converted to Censor objects
        """
        self.censors = [
            Censor(**censor) if isinstance(censor, dict) else censor
            for censor in self.censors
        ]


@dataclass
class PartInformationConfig:
    enabled_parts: list[str] = field(default_factory=list)

    parts_settings: dict[str, PartSettingsConfig] = field(default_factory=dict)
    merge_settings: MergingConfig = field(default_factory=MergingConfig)


# Config Proper
@dataclass(slots=True)
class Config:
    # File Information
    dev_settings: DevConfig
    file_settings: FileConfig

    # Program Settings
    image_settings: ImageConfig
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
        image_settings = config_data.get("image_settings", {})
        video_settings = config_data.get("video_settings", {})
        render_settings = config_data.get("render_settings", {})
        ai_settings = config_data.get("ai_settings", {})
        censor_settings = config_data.get("censor_settings", {})

        # Censor Part Information
        enabled_parts = censor_settings.get("enabled_parts", [])
        default_part_settings = censor_settings.get("default_part_settings", {})
        reverse_censor_settings = censor_settings.get("reverse_censor_settings", {})
        merge_settings = censor_settings.get("merge_settings", {})

        # Extract part-specific settings
        parts_settings = {}
        default_settings_object = PartSettingsConfig(**default_part_settings)

        # # Handle Part Data
        for part_name in enabled_parts:
            part_data = censor_settings.get(part_name, {})

            if (
                not part_data.get("censors")
                and part_data.get("state", "").lower() == "revealed"
            ):
                part_data["censors"] = []

            # Update Part Data
            part_object = replace(default_settings_object, **part_data)
            part_object.__post_init__()
            parts_settings[part_name] = part_object

        # Compile
        return {
            "dev_settings": DevConfig(**dev_settings),
            "file_settings": FileConfig(**file_settings),
            "image_settings": ImageConfig(**image_settings),
            "video_settings": VideoConfig(**video_settings),
            "rendering_settings": RenderingConfig(**render_settings),
            "ai_settings": AIConfig(**ai_settings),
            "default_censor_settings": default_settings_object,
            "censor_settings": PartInformationConfig(
                enabled_parts=enabled_parts,
                parts_settings=parts_settings,
                merge_settings=MergingConfig(**merge_settings),
            ),
            "reverse_censor": ReverseCensorConfig(censors=reverse_censor_settings),
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
        full_config_path = os.path.join(CONFIGS_FOLDER, "defaults", config_path)

        # If it doesn't Exist, Check for Local Paths
        if not os.path.exists(full_config_path):
            full_config_path = os.path.join(main_file_path, config_path)

        # If It still doesn't Exist, It must be Wrong
        if not os.path.exists(full_config_path):
            raise FileNotFoundError(
                f"Cannot find config at {config_path} or {full_config_path}"
            )

        # Load Config File
        with open(full_config_path, "r") as file:
            config_data = yaml.safe_load(file) or {}

        return cls(**Config._process_dict_data(config_data))

    @classmethod
    def from_dictionary(cls, dict_config: dict[str, Any]) -> "Config":
        return cls(**Config._process_dict_data(dict_config))
