from pathlib import Path

import pytest

from tests.utils import (
    run_image_test,
)


def list_config_files():
    config_folder = Path("src") / "censor_engine" / "libs" / "configs"
    return [
        str(Path(config).relative_to(config_folder))
        for config in list(config_folder.glob("*.yml"))
    ]


config_files = list_config_files()

expected_error: dict[str, float] = {
    "09_red_tape.yml": 2,
    "11_trigger_focus.yml": 3,
    "10_state_approved_triggers.yml": 4,
    "zz_dev.yml": 5,
    "raw_data.yml": 3.5,
}


@pytest.mark.parametrize("config_name", config_files)
def test_core_configs(config_name, dummy_input_image_data) -> None:
    error = expected_error.get(config_name)
    error_arg = {"mean_absolute_error": error} if error else {}
    run_image_test(
        dummy_input_image_data,
        config=config_name,
        subfolder=config_name,
        batch_tests=False,
        expect_png=config_name == "06_cutouts.yml",
        **error_arg,  # type: ignore
    )
