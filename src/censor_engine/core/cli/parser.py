import argparse

from censor_engine.core_engine.tools.debugger import DebugLevels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="CensorEngine",
        description="Censors Images and Videos",
    )

    # =========================
    # Standard Arguments
    # =========================

    parser.add_argument(
        "--uncensored-location",
        type=str,
        help="Path to uncensored files",
    )

    parser.add_argument(
        "--config-location",
        type=str,
        help="Path to config YAML file",
    )

    parser.add_argument(
        "--debug-level",
        type=str,
        choices=[level.name.lower() for level in DebugLevels],
        help="Debug verbosity level",
    )

    # =========================
    # Feature Flags
    # =========================

    parser.add_argument(
        "-sm",
        "--show-stat-metrics",
        action="store_true",
        help="Display processing statistics",
    )

    parser.add_argument(
        "-pi",
        "--pad-individual-items",
        action="store_true",
        help="Pad censored items individually",
    )

    parser.add_argument(
        "-dt",
        "--dev-tools",
        action="store_true",
        help="Enable developer tools",
    )

    parser.add_argument(
        "-fo",
        "--show-full-output-path",
        action="store_true",
        help="Display full output paths",
    )

    parser.add_argument(
        "-td",
        "--using-test-data",
        action="store_true",
        help="Use internal test data",
    )

    parser.add_argument(
        "-example",
        "--example-preview",
        action="store_true",
        help="Run preview example mode",
    )

    return parser
