from dataclasses import dataclass, field
from pathlib import Path

from censor_engine._typing import Image
from censor_engine.core_engine.cli.structs import (
    EngineFlags,
    ParsedArguments,
)
from censor_engine.models.core.tools.debugger.enums import DebugLevelfrom censor_engine.models.config import Config
from censor_engine.models.lib_models.detectors.ai_models import AIModel
from censor_engine.paths._OLD.base import PathManager


@dataclass
class EngineSettings:
    config: Config

    uncensored_location_override: Path | None = None
    debug_level: DebugLevel = DebugLevel.NONE
    flags: EngineFlags = field(default_factory=EngineFlags)

    def load_parsed_args(self, args: ParsedArguments) -> None:
        self.config = args.config or self.config
        self.uncensored_location_override = (
            args.uncensored_location_override
            or self.uncensored_location_override
        )
        self.debug_level = args.debug_level or self.debug_level

        self.flags = args.flags


@dataclass(slots=True)
class PipelineContext:
    config: Config
    debug_level: DebugLevel
    flags: EngineFlags
    path_manager: PathManager
    live_detectors: list[AIModel]
    test_mode: bool


@dataclass(slots=True)
class EngineResult:
    images: list[Image]
    videos_processed: int
    durations: list[float]
    detection_count: int
