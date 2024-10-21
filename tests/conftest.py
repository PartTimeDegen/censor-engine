import os
import pytest

from censorengine.backend.handlers.file_handlers.config_handler import (
    CONFIG,
    load_config,
)

DEBUG = True


@pytest.fixture(scope="function", autouse=True)
def setup_config():
    def get_config(config_path, results_path):
        # Get Config
        file_path = os.path.dirname(__file__)
        list_path = file_path.split(os.sep)
        list_path = list_path[: list_path.index("tests")]
        root_path = os.sep.join([*list_path, "src"])
        complete_config_path = os.path.join(
            os.sep.join(list_path),
            "tests",
            "configs",
            config_path,
        )
        load_config(complete_config_path)

        # Get results folder
        uncensored_test_folder = os.path.join(
            "aaa_tests",
            results_path,
        )
        intended_results_folder = os.path.join(
            os.sep.join(list_path),
            "tests",
            "indended_results",
            results_path,
        )
        CONFIG.update({"image_folder": uncensored_test_folder})
        CONFIG.update({"debug_mode": DEBUG})

        output = {
            "root_path": root_path,
            "config_path": complete_config_path,
            "results_path": intended_results_folder,
            "image_folder": CONFIG["image_folder"],
        }
        print()
        print()
        print("debug log fixture")
        for key, value in output.items():
            print(f"{key}: {value}")
        print()
        print()

        return output

    return get_config
