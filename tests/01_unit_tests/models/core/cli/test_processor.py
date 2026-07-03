from unittest.mock import patch

from censor_engine.models.core.cli._parser import build_parser
from censor_engine.models.core.cli._processor import process_arguments
from censor_engine.models.core.tools.debugger.enums import DebugLevel


def parse_args(*args):
    return build_parser().parse_args(args)


class TestDefaults:
    def test_defaults(self):
        parsed = process_arguments(parse_args())

        assert parsed.uncensored_location_override is None
        assert parsed.config is None
        assert parsed.debug_level == DebugLevel.NONE

        flags = parsed.flags
        assert flags.show_stat_metrics is False
        assert flags.pad_individual_items is False
        assert flags.dev_tools is False
        assert flags.show_full_output_path is False
        assert flags.using_test_data is False
        assert flags.example_preview is False
        assert flags.using_shortcut is False


class TestUncensoredLocation:
    def test_path_is_converted(self):
        parsed = process_arguments(
            parse_args("--uncensored-location", "/tmp/input")
        )

        assert str(parsed.uncensored_location_override) == "/tmp/input"

    def test_relative_path_sets_shortcut(self):
        parsed = process_arguments(
            parse_args("--uncensored-location", "./data")
        )

        assert parsed.flags.using_shortcut is True

    def test_absolute_path_does_not_set_shortcut(self):
        parsed = process_arguments(
            parse_args("--uncensored-location", "/tmp/data")
        )

        assert parsed.flags.using_shortcut is False


class TestConfig:
    @patch("censor_engine.models.libraries.configs.config.Config.from_yaml")
    def test_config_loaded(self, mock_from_yaml):
        config = object()
        mock_from_yaml.return_value = config

        parsed = process_arguments(
            parse_args("--config-location", "config.yml")
        )

        mock_from_yaml.assert_called_once_with("config.yml")
        assert parsed.config is config


class TestDebugLevel:
    def test_debug_level_is_parsed(self):
        parsed = process_arguments(parse_args("--debug-level", "full"))

        assert parsed.debug_level == DebugLevel.FULL


class TestFlags:
    def test_all_flags_are_copied(self):
        parsed = process_arguments(
            parse_args(
                "--show-stat-metrics",
                "--pad-individual-items",
                "--dev-tools",
                "--show-full-output-path",
                "--using-test-data",
                "--example-preview",
            )
        )

        flags = parsed.flags

        assert flags.show_stat_metrics
        assert flags.pad_individual_items
        assert flags.dev_tools
        assert flags.show_full_output_path
        assert flags.using_test_data
        assert flags.example_preview
