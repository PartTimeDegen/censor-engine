from dataclasses import dataclass, field, fields
import os
from typing import Any, Optional

import yaml
from censorengine.backend.constants.files import CONFIGS_FOLDER
from censorengine.backend.models.detected_part import PartState, Censor
from censorengine.backend.constants.exceptions import (
    MissingEssentialConfigSection,
)
from types import MappingProxyType


class _ConfigPart:
    name: str
    minimum_score: Optional[float] = None
    censors: Optional[list[Censor]] = None
    shape: Optional[str] = None
    margin: Optional[int | float | dict[str, float]] = None
    state: Optional[PartState] = None
    protected_shape: Optional[str] = None
    use_global_area: Optional[bool] = None

    internal_defaults = MappingProxyType(
        {
            "minimum_score": 0.20,
            "censors": [{"function": "null", "args": {}}],
            "shape": "box",
            "margin": 0.0,
            "state": PartState.UNPROTECTED,
            "protected_shape": None,
            "use_global_area": True,
        }
    )

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    @staticmethod
    def _get_else_default(section: dict[str, Any], name: str) -> Any:
        if value := section.get(name):
            return value
        else:
            try:
                return _ConfigPart.internal_defaults[name]
            except KeyError:
                raise KeyError(f"Default for '{name}' has not been set")

    @staticmethod
    def load_censors(
        found_censors: Optional[list[dict]] = None,
    ) -> list[Censor]:
        if not found_censors:
            return []

        censor_list = []

        for censor in found_censors:
            censor_part = Censor(censor["function"], censor.get("args", {}))

            censor_list.append(censor_part)

        return censor_list

    @staticmethod
    def load_state(found_state: str) -> PartState:
        match found_state:
            case "protected":
                new_state = PartState.PROTECTED
            case "revealed":
                new_state = PartState.REVEALED
            case "unprotected":
                new_state = PartState.UNPROTECTED
            case _:
                new_state = PartState.UNPROTECTED

        return new_state

    @staticmethod
    def set_part_defaults(config_info: dict[str, Any]) -> None:
        default_section = config_info.get("defaults")
        if not default_section:
            default_section = {}

        _ConfigPart.minimum_score = _ConfigPart._get_else_default(
            default_section, "minimum_score"
        )
        _ConfigPart.censors = _ConfigPart.load_censors(
            _ConfigPart._get_else_default(default_section, "censors")
        )
        _ConfigPart.shape = _ConfigPart._get_else_default(default_section, "shape")
        _ConfigPart.margin = _ConfigPart._get_else_default(default_section, "margin")
        _ConfigPart.state = _ConfigPart.load_state(
            _ConfigPart._get_else_default(default_section, "state")
        )
        _ConfigPart.protected_shape = _ConfigPart._get_else_default(
            default_section, "minimum_score"
        )
        _ConfigPart.protected_shape = _ConfigPart._get_else_default(
            default_section, "use_global_area"
        )

    def set_part_fields(self, config_info: dict[str, Any]) -> None:
        part_section = config_info.get(self.name)
        if not part_section:
            return

        def get_part(
            self: _ConfigPart,
            part_section: dict[str, Any],
            name: str,
        ) -> Any:
            subsection = part_section.get(name, "missing")

            # Part Remains Unchanged
            if subsection == "missing":
                return getattr(self, name)

            # Part has Section
            # Handle Special Cases
            # # State's Enum
            if name == "state":
                subsection = self.load_state(subsection)

            # Censors Object from Dict
            if name == "censors":
                subsection = self.load_censors(subsection)

            return subsection

        self.minimum_score = get_part(self, part_section, "minimum_score")
        self.censors = get_part(self, part_section, "censors")
        self.shape = get_part(self, part_section, "shape")
        self.margin = get_part(self, part_section, "margin")
        self.state = get_part(self, part_section, "state")
        self.protected_shape = get_part(self, part_section, "protected_shape")
        self.use_global_area = get_part(self, part_section, "use_global_area")


class ReverseCensorPart:
    censors: list[Censor]


@dataclass
class Config:
    # Raw Config
    config_file: dict[str, Any] = field(default_factory=dict)

    # Files
    file_prefix: str = field(default="")
    file_suffix: str = field(default="")
    force_png: bool = field(default=False)

    # Folders
    uncensored_folder: str = field(default="00_uncensored")
    censored_folder: str = field(default="01_censored")

    # Debug
    debug_level: int = field(default=0)
    debug_images: bool = field(default=False)

    # File Additions
    parts_enabled: list[str] = field(default_factory=list)

    # Merging
    merge_enabled: bool = field(default=False)
    merge_range: int | float = field(default=-1)
    merge_groups: list[list[str]] = field(default_factory=list)

    # Information
    # # Smoothing
    enable_smoothing: bool = field(default=False)

    # # Defaults
    part_settings: dict[str, _ConfigPart] = field(default_factory=dict)

    # # Reverse Censor
    reverse_censor_enabled: bool = field(default=False)
    reverse_censors: list[Censor] = field(default_factory=list)

    def __init__(self, main_file_path, config_path):
        # Required for mutatable fields
        self.config_file = {}
        self.parts_enabled = []
        self.merge_groups = []
        self.part_settings = {}
        self.reverse_censors = []

        self._load_config(main_file_path, config_path)
        self._generate_fields()
        self._set_fields()

    def _generate_fields(self):
        for found_field in fields(self):
            if found_field.name not in vars(self):
                setattr(
                    self,
                    found_field.name,
                    getattr(self, found_field.name),
                )

    def _load_config(self, main_file_path, config_path):
        # Find Config File
        full_config_path = os.path.join(
            CONFIGS_FOLDER,
            "defaults",
            config_path,
        )
        print(full_config_path)
        if not os.path.exists(full_config_path):
            full_config_path = os.path.join(
                main_file_path,
                config_path,
            )
        if not os.path.exists(full_config_path):
            raise FileNotFoundError(
                f"Cannot find config at {config_path} or {full_config_path}"
            )

        # Load Config File
        with open(full_config_path) as stream:
            try:
                # Check if Can Access Data
                config_data = yaml.safe_load(stream)

                # Reset Variable
                self.config_file = {}

                # Update Variable
                self.config_file.update(**config_data)

            except yaml.YAMLError as exc:
                raise ValueError(exc)
        return full_config_path

    def _set_fields(self):
        config_file = self.config_file
        manual_attributes = [
            "config",
            "config_file",
            "merge_enabled",
            "merge_range",
            "merge_groups",
            "enable_smoothing",
            "part_settings",
            "reverse_censor_enabled",
        ]

        attributes = [
            attribute for attribute in vars(self) if attribute not in manual_attributes
        ]

        # Root Level Attributes
        for attribute in attributes:
            if config_attribute := config_file.get(attribute):
                setattr(self, attribute, config_attribute)

        # Merging
        if config_merge := config_file.get("merging"):
            self.merge_enabled = True
            if merge_range := config_merge.get("merge_range"):
                self.merge_range = merge_range
            if merge_groups := config_merge.get("merge_groups"):
                self.merge_groups = merge_groups

        # Rendering
        if config_rendering := config_file.get("rendering"):
            if config_smoothness := config_rendering.get("mask_smoothing"):
                self.enable_smoothing = config_smoothness

        # Information
        if config_info := config_file.get("information"):
            self._build_info_settings(config_info)
        else:
            raise MissingEssentialConfigSection

    def _build_info_settings(self, config_info):
        # Set Defaults
        _ConfigPart.set_part_defaults(config_info)

        # Set Parts
        for part_name in self.parts_enabled:
            self.part_settings[part_name] = _ConfigPart(name=part_name)
            self.part_settings[part_name].set_part_fields(config_info)

        # Reverse Censor
        if config_section := config_info.get("reverse_censors"):
            self.reverse_censor_enabled = True
            self.reverse_censors = _ConfigPart.load_censors(config_section)
        else:
            self.part_settings["reverse_censors"] = None

    def dev_generate_new_parts(self):
        for part in self.parts_enabled:
            if not self.part_settings[part]:
                self.part_settings[part] = _ConfigPart(name=part)

    def dev_load_censors(self, part, list_of_censors):
        self.part_settings[part].censors = _ConfigPart.load_censors(list_of_censors)
