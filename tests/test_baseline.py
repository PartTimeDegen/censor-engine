from pathlib import Path
import cv2
import numpy as np
from tests.test_data.utils import dummy_input_image, assert_image


def test_utils(dummy_input_image):
    input_file_path, image_generator = dummy_input_image

    input_file = cv2.imread(input_file_path)
    cv2.imwrite("output.jpg", input_file)

    assert_image(input_file, np.zeros((100, 100, 3), dtype=np.uint8), "tests")
