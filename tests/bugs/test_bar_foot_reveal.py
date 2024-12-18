import os
from censorengine import CensorEngine  # type: ignore
from censorengine.backend._dev import assert_files_are_intended  # type: ignore


# @pytest.mark.timeout(5)
def test_shapes(root_path):
    # TODO: This only happened when the merge groups weren't applied, do this then find issue
    # Test Type
    test_loc = os.path.join("bugs/bar_foot_reveal")

    # Initiate
    ce = CensorEngine(
        root_path,
        config="000_tests/03_configs/bugs/bar_foot_reveal/red_tape.yml",
        test_mode=True,
        debug_mode=True,
    )

    folder_uncen = os.path.join("000_tests/00_uncensored", test_loc)
    folder_cen = "000_tests/01_censored"

    # Set Config
    ce.config.uncensored_folder = folder_uncen
    ce.config.censored_folder = folder_cen

    # Start CensorEngine
    ce.start()

    assert_files_are_intended(root_path, test_loc)
