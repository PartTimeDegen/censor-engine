from argparse import Namespace
from pathlib import Path

from censor_engine.models.core.tools.debugger.enums import DebugLevel
from censor_engine.models.libraries.configs.config import Config

from .structs import (
    EngineFlags,
    ParsedArguments,
)


def process_arguments(args: Namespace) -> ParsedArguments:
    """
    This function processes the arguments and contains the logic for the
    argument system.

    Args:
        args: Arguments object

    Returns:
        Parsed Parsed Arguments Structure that contains the info

    """
    # Uncensored Location
    uncen_location_override = (
        Path(args.uncensored_location) if args.uncensored_location else None
    )

    # Config
    config = None
    if args.config_location:
        config = Config.from_yaml(args.config_location)

    # Debug Level
    debug_level = DebugLevel.NONE
    if args.debug_level:
        debug_level = DebugLevel[args.debug_level.upper()]

    # Flags
    flags = EngineFlags(
        show_stat_metrics=args.show_stat_metrics,
        pad_individual_items=args.pad_individual_items,
        dev_tools=args.dev_tools,
        show_full_output_path=args.show_full_output_path,
        using_test_data=args.using_test_data,
        example_preview=args.example_preview,
    )

    if uncen_location_override and args.uncensored_location[0] == ".":
        flags.using_shortcut = True

    return ParsedArguments(
        uncensored_location_override=uncen_location_override,
        debug_level=debug_level,
        config=config,
        flags=flags,
    )
