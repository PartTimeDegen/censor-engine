from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.helpers._config import IGNORE_CHECKME


def _get_test_data_path(file_path: Path) -> Path:
    # Get Current Path
    current_file = file_path.relative_to(Path.cwd())

    # Convert Folder to Test Data
    test_data_file = list(current_file.parts)
    test_data_file[1] = str(Path("00_test_data") / test_data_file[1])
    test_data_file[-1] = test_data_file[-1].strip(".py")
    return Path(*test_data_file)


def _create_new_test_data(
    expected_file: Path, checkme_file: Path, test_folder: Path, image
):
    # Create Expected.jpg
    cv2.imwrite(str(expected_file), image)

    # Create Check File
    if not IGNORE_CHECKME:
        checkme_file.touch()

    # Check
    msg = (
        "Expected Image Data didn't exist, please check, if it's good, "
        f"delete the .checkme file. ({test_folder})"
    )
    pytest.fail(msg)


def run_comparison_test(test_folder, expected_image, image):
    THRESHOLD = 1

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    diff_pixels = np.count_nonzero(expected_image != image)

    total_pixels = expected_image.shape[0] * expected_image.shape[1]
    diff_percent = (diff_pixels / total_pixels) * 100

    test = diff_percent < THRESHOLD
    if not test:
        cv2.imwrite(
            str(test_folder / "diff.png"),
            cv2.absdiff(expected_image, image),
        )

    assert test


def _check_files(expected_file: Path, test_folder: Path, image):
    # Save Current Version
    output_file = test_folder / "output.png"
    cv2.imwrite(str(output_file), image)

    # Compare Images
    expected_image = cv2.imread(str(expected_file))
    run_comparison_test(test_folder, expected_image, image)


def handle_test_data(test_name: str, image: np.ndarray, file_path: Path):
    # Make Base Folders
    test_folder = _get_test_data_path(file_path) / test_name
    test_folder.mkdir(parents=True, exist_ok=True)

    # Check if Exists
    expected_file = test_folder / "expected.png"
    checkme_file = test_folder / ".checkme"

    if not expected_file.exists():
        _create_new_test_data(expected_file, checkme_file, test_folder, image)

    if checkme_file.exists() and not IGNORE_CHECKME:
        cv2.imwrite(str(expected_file), image)
        msg = (
            "Found the .checkme file, "
            f"please check and delete. ({test_folder})"
        )
        pytest.fail(msg)

    else:
        _check_files(expected_file, test_folder, image)
        if IGNORE_CHECKME:
            pytest.fail("Disable Checkme Bypass")
