import datetime
import os
import shutil

import cv2
import numpy as np

from .handlers.file_handlers.config_handler import CONFIG

DEV_IS_FIRST_HIT = True


def dev_get_mask(part, mask, identifier: str = ""):
    if CONFIG["debug_mode"]:
        return cv2.imwrite(
            f"censorengine/censored/images/.dev/{part['class']}_{identifier}.jpg",
            mask,
        )


def check(func, part, mask):
    def wrapper():
        dev_get_mask(part, mask, "1_before")
        func()
        dev_get_mask(part, mask, "2_after")

    return wrapper


def dev_decompose_mask(dict_info, image, part=None, prefix="", suffix=""):
    """
    # import backend.dev as dev
    # dev.dev_decompose_mask(dict_info, mask, part, prefix="overlaps", suffix=part["class"])

    """

    if not CONFIG["debug_mode"]:
        return

    if not os.path.exists(".dev"):
        os.makedirs(".dev")

    global DEV_IS_FIRST_HIT

    if DEV_IS_FIRST_HIT:
        backup_folder = os.path.join(
            ".dev-backups",
            f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}",
        )
        os.makedirs(
            backup_folder,
            exist_ok=True,
        )
        for file_name in os.listdir(".dev"):
            shutil.move(os.path.join(".dev", file_name), backup_folder)

        DEV_IS_FIRST_HIT = False

    DEV_FOLDERS = {
        folder[folder.find("-") + 1 :]: int(folder[: folder.find("-")])
        for folder in os.listdir(".dev")
    }

    # Find Index, update if name is different
    if not os.path.exists(".dev"):
        os.makedirs(".dev")

    # Get Folder
    folders = os.listdir(".dev")

    # Get Index Info
    index = [int(folder[: folder.find("-")]) for folder in folders]
    if index:
        max_index = max(index)
    else:
        max_index = 0
    new_index = max_index + 1

    if DEV_FOLDERS.get(prefix):
        prefix = f"{DEV_FOLDERS[prefix]:02d}-{prefix}"
    else:
        DEV_FOLDERS[prefix] = new_index
        prefix = f"{new_index:02d}-{prefix}"

    name = [
        word for word in [prefix, dict_info["file_image_name"]] if word != ""
    ]
    if part is not None:
        part_name = part["class"]
        folder_name = os.path.join(".dev", *name)
    else:
        part_name = "image"
        folder_name = os.path.join(".dev", "misc", *name)

    os.makedirs(folder_name, exist_ok=True)

    if suffix == "":
        file_name = f"{folder_name}/{part_name}.jpg"
    else:
        file_name = f"{folder_name}/{part_name}-{suffix}.jpg"

    cv2.imwrite(file_name, image)


def dev_compare_before_after_if_different(
    root_path,
    results_path,
    file_name="test.jpg",
):
    DIFFERENCE_THRESHOLD = 20 * 20

    # Get Files
    loc_file_uncensored = os.path.join(
        root_path,
        "01_censored",
        "aaa_tests",
        results_path,
        file_name,
    )
    results_folder_path = os.path.join(
        os.getcwd(),
        "tests",
        "intended_results",
        results_path,
    )
    loc_file_censored = os.path.join(results_folder_path, file_name)

    print(loc_file_uncensored)
    print(loc_file_censored)

    before = cv2.imread(loc_file_uncensored)
    after = cv2.imread(loc_file_censored)

    # Check for Bad Reads
    if before is None:
        raise FileNotFoundError(loc_file_uncensored)
    if after is None:
        assert False, "No Proper Result Found"

    image_difference = np.bitwise_xor(before, after)

    # Denoise Image
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_difference = cv2.cvtColor(image_difference, cv2.COLOR_BGR2GRAY)
    image_difference[np.where(image_difference > 128)] = 255
    image_difference = cv2.erode(image_difference, element, iterations=2)
    image_difference = cv2.divide(
        cv2.GaussianBlur(image_difference, (0, 0), sigmaX=11, sigmaY=11),
        image_difference,
        scale=255,
    )
    image_difference = cv2.erode(image_difference, element, iterations=2)
    image_difference[np.where(image_difference > 0)] = 255
    image_difference = cv2.morphologyEx(
        image_difference, cv2.MORPH_CLOSE, element
    )

    # Get Percent of Image that Isn't Black
    count_white = np.sum(image_difference == 255)

    cv2.imwrite(
        os.path.join(results_folder_path, f"difference_{file_name}.jpg"),
        image_difference,
    )
    if count_white > DIFFERENCE_THRESHOLD:
        return True

    return False


def assert_files_are_intended(root_path, results_path):
    pass_files = True
    failed_files = []

    # Assertion
    for file_seen in os.listdir(
        os.path.join(
            root_path,
            "01_censored",
            "aaa_tests",
            results_path,
        )
    ):

        result = dev_compare_before_after_if_different(
            root_path,
            results_path,
            file_name=file_seen,
        )
        if result == True:
            pass_files = False
            failed_files.append(file_seen)

    assert pass_files, failed_files
