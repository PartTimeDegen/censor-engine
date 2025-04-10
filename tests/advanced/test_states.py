import os
import pytest
from censorengine import CensorEngine  # type: ignore
from censorengine.backend._dev import assert_files_are_intended  # type: ignore

state_combinations = {
    "unprotected/protected": "000_tests/03_configs/advanced/states/01_unprotected_on_protected.yml",
    "unprotected/revealed": "000_tests/03_configs/advanced/states/01_unprotected_on_revealed.yml",
    "unprotected/unprotected": "000_tests/03_configs/advanced/states/01_unprotected_on_unprotected.yml",
    "revealed/protected": "000_tests/03_configs/advanced/states/02_revealed_on_protected.yml",
    "revealed/revealed": "000_tests/03_configs/advanced/states/02_revealed_on_revealed.yml",
    "revealed/unprotected": "000_tests/03_configs/advanced/states/02_revealed_on_unprotected.yml",
    "protected/protected": "000_tests/03_configs/advanced/states/03_protected_on_protected.yml",
    "protected/revealed": "000_tests/03_configs/advanced/states/03_protected_on_revealed.yml",
    "protected/unprotected": "000_tests/03_configs/advanced/states/03_protected_on_unprotected.yml",
}  # TODO: I'm sure there's a nicer way to do this


@pytest.mark.parametrize("state_config", list(state_combinations.keys()))
def test_states(state_config, root_path):
    # NOTE: Exposed parts are target, rest is other [PINK/BLACK]

    # Test Type
    test_loc = os.path.join("advanced/states", state_config)

    # Initiate
    ce = CensorEngine(
        root_path,
        config_data=state_combinations[state_config],
        test_mode=True,
    )

    folder_uncen = os.path.join("000_tests/00_uncensored", test_loc)
    folder_cen = "000_tests/01_censored"

    # Set Config
    parts = [
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "FEMALE_GENITALIA_EXPOSED",
        "FEMALE_GENITALIA_COVERED",
        "BUTTOCKS_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
    ]

    ce._config.file_settings.uncensored_folder = folder_uncen
    ce._config.file_settings.censored_folder = folder_cen
    ce._config.censor_settings.enabled_parts = parts
    # ce.config.dev_generate_new_parts()

    # assert False, [
    #     f"{part} : {ce.config.censor_settings.parts_settings[part].state}" for part in parts
    # ]

    # Start CensorEngine
    ce.start()

    assert_files_are_intended(root_path, test_loc)
