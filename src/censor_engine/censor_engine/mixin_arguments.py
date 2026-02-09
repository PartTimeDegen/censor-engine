import argparse
from pathlib import Path
from typing import Any

from censor_engine.censor_engine.tools.debugger import DebugLevels
from censor_engine.models.config import Config
from censor_engine.models.structs import Mixin


class MixinArguments(Mixin):
    def load_config(
        self,
        base_folder: Path,
        config_data: str | dict[str, Any],
    ) -> Config:
        if isinstance(config_data, str):
            return Config.from_yaml(base_folder, config_data)
        if isinstance(config_data, dict):
            return Config.from_dictionary(config_data)
        msg = "invalid type used"
        raise TypeError(msg)

    def _parse_arguments(
        self,
        base_folder: Path,
        config_data: str | dict[str, Any],
        output_dict: dict[str, Any],
    ) -> dict[str, Any]:
        # Parser
        parser = argparse.ArgumentParser(
            prog="CensorEngine",
            description="Censors Images",
        )
        arg_mapper = {
            "uncensored_location": "uncensored-location",
            "config_location": "config-location",
            "debug_level": "debug-level",
        }

        flag_mapper = {
            "show_stat_metrics": "sm",
            "pad_individual_items": "pi",
            "dev_tools": "dt",
            "show_full_output_path": "fo",
            "using_test_data": "td",
            "example_preview": "example",
        }

        # Add Args
        for value in arg_mapper.values():
            parser.add_argument(f"--{value}", action="store")

        # Add Flags
        for long_flag_name, short_flag_name in flag_mapper.items():
            parser.add_argument(
                f"-{short_flag_name}",
                f"--{long_flag_name.replace('_', '-')}",
                dest=long_flag_name,
                action="store_true",
                help=f"Enable {long_flag_name.replace('_', ' ')}",
            )

        # Collect Args
        args, _ = parser.parse_known_args()

        # Handle Args
        # # File Location
        if loc := args.uncensored_location:
            output_dict["arg_loc"] = Path(loc)

        # # Config Location
        if config := args.config_location:
            config_data = config
        output_dict["config"] = self.load_config(base_folder, config_data)

        # # Debug Information
        if debug_word := args.debug_level:
            try:
                output_dict["debug_level"] = DebugLevels[debug_word.upper()]
                print(f"**Found Debug Level = {output_dict['debug_level']}**")
            except ValueError:
                msg = f"Invalid DebugLevels value: {debug_word!s}"
                raise ValueError(msg)

        # Handle Handle Flags
        output_dict["flags"] = {key: getattr(args, key) for key in flag_mapper}
        for value in output_dict["flags"].values():
            if value:
                pass

        if not output_dict["config"]:
            output_dict["config"] = self.load_config(base_folder, config_data)

        if output_dict["arg_loc"] and (
            output_dict["arg_loc"].parts
            and output_dict["arg_loc"].parts[0] == "."
        ):
            output_dict["flags"]["_using_shortcut"] = True

        return output_dict
