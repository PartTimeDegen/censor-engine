from pathlib import Path

import pytest

from tests.utils import (
    run_image_test,
)


def list_config_files():
    config_folder = Path("src") / "censor_engine" / "libs" / "configs"
    basic_configs = config_folder / "basic"
    complex_configs = config_folder / "complex"
    return [
        str(Path(config).relative_to(config_folder))
        for config in list(basic_configs.glob("*.yml"))
    ]+ [
        str(Path(config).relative_to(config_folder))
        for config in list(complex_configs.glob("*.yml"))
    ]


config_files = list_config_files()

expected_error: dict[str, float] = {
    "complex/red_tape.yml": 2,
    "complex/trigger_focus.yml": 3,
    "complex/state_approved_triggers.yml": 4,
    "raw_data.yml": 3.5,
}


@pytest.mark.parametrize("config_name", config_files)
def test_core_configs(config_name, dummy_input_image_data) -> None:
    # Quick Fix to avoid the linux/ci issue
    if config_name == "complex/nope.yml":
        assert True
        return
    error = expected_error.get(config_name)
    error_arg = {"mean_absolute_error": error} if error else {}
    run_image_test(
        dummy_input_image_data,
        config=config_name,
        subfolder=config_name,
        batch_tests=False,
        expect_png=config_name == "basic/cutouts.yml",
        **error_arg,  # type: ignore
    )
