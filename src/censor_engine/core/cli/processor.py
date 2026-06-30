from argparse import Namespace
from pathlib import Path

from censor_engine.core_engine.tools.debugger import DebugLevels
from censor_engine.core_engine.utils import load_config

from .structs import (
    EngineFlags,
    ParsedArguments,
)


def process_arguments(
    args: Namespace,
    base_folder: Path,
) -> ParsedArguments:
    # Uncensored Location
    uncen_location_override = (
        Path(args.uncensored_location) if args.uncensored_location else None
    )

    # Config
    config = None
    if args.config_location:
        config = load_config(
            base_folder,
            args.config_location,
        )

    # Debug Level
    debug_level = DebugLevels.NONE
    if args.debug_level:
        debug_level = DebugLevels[args.debug_level.upper()]

    # Flags
    flags = EngineFlags(
        show_stat_metrics=args.show_stat_metrics,
        pad_individual_items=args.pad_individual_items,
        dev_tools=args.dev_tools,
        show_full_output_path=args.show_full_output_path,
        using_test_data=args.using_test_data,
        example_preview=args.example_preview,
    )

    if (
        uncen_location_override
        and uncen_location_override.parts
        and uncen_location_override.parts[0] == "."
    ):
        flags.using_shortcut = True

    return ParsedArguments(
        uncensored_location_override=uncen_location_override,
        debug_level=debug_level,
        config=config,
        flags=flags,
    )
