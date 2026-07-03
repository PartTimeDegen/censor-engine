from dataclasses import dataclass
from pathlib import Path

from censor_engine.models.core.tools.debugger.enums import DebugLevel
from censor_engine.models.libraries.configs.config import Config


@dataclass(slots=True)
class EngineFlags:
    show_stat_metrics: bool = False
    pad_individual_items: bool = False
    dev_tools: bool = False
    show_full_output_path: bool = False
    using_test_data: bool = False
    example_preview: bool = False
    using_shortcut: bool = False


@dataclass(slots=True)
class ParsedArguments:
    flags: EngineFlags
    uncensored_location_override: Path | None
    debug_level: DebugLevel
    config: Config | None
