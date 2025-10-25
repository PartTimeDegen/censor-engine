from pathlib import Path

import cv2

from tests.input_image import ImageGenerator
from tests.input_video import VideoGenerator, MotionInformation, MotionTrack
from tests.utils import (
    load_config_base_yaml,
    run_image_test,
    run_video_test,
)


def test_test_working(dummy_input_image_data) -> None:
    config = load_config_base_yaml()

    run_image_test(dummy_input_image_data, config=config)


def test_image_generator() -> None:
    ig = ImageGenerator(Path())
    cv2.imwrite("example.jpg", ig.make_test_image())

    assert True

def test_video_generator() -> None:
    vg = VideoGenerator(Path(), "test_video.mp4")
    movement = MotionTrack(1, (-1, 0))
    list_of_movements = [
        MotionInformation("FEMALE_BREAST_EXPOSED", ("FEMALE_BREAST_EXPOSED", 140, "left"), movement),
        MotionInformation("FEMALE_BREAST_EXPOSED", ("FEMALE_BREAST_EXPOSED", 140, "right"), movement),
    ]
    

    vg.make_test_video(list_of_movements)

    assert True

def test_video_working(dummy_input_video_data) -> None:
    config = load_config_base_yaml()
    
    movement = MotionTrack(1, (-1, 0))
    list_of_movements = [
        MotionInformation("FEMALE_BREAST_EXPOSED", ("FEMALE_BREAST_EXPOSED", 140, "left"), movement),
        MotionInformation("FEMALE_BREAST_EXPOSED", ("FEMALE_BREAST_EXPOSED", 140, "right"), movement),
    ]

    run_video_test(dummy_input_video_data(list_of_movements), config=config)

    
