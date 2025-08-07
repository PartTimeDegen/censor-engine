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


@pytest.mark.parametrize("config_name", config_files)
def test_core_configs(config_name, dummy_input_image_data):
    run_image_test(
        dummy_input_image_data,
        config_name,
        subfolder=config_name,
        batch_tests=False,
        expect_png=True if config_name == "06_cutouts.yml" else False,
    )
