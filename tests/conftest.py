from collections.abc import Generator

import cv2
import pytest

from tests.input_image import ImageGenerator
from tests.input_video import VideoGenerator
from tests.utils import ImageFixtureData, VideoFixtureData


@pytest.fixture
def dummy_input_image_data(tmp_path) -> Generator[ImageFixtureData]:
    input_path = tmp_path / "input.jpg"

    image_generator = ImageGenerator(input_path)
    dummy_img = image_generator.make_test_image()

    cv2.imwrite(str(input_path), dummy_img)
    return ImageFixtureData(
        path=input_path,
        generator=image_generator,
        parts=image_generator.parts,
    )  # type: ignore


@pytest.fixture
def dummy_input_video_data(tmp_path):
    def _make_fixture(input_data: list):
        input_path = tmp_path
        video_generator = VideoGenerator(input_path, "input_video.mp4")

        video_generator.make_test_video(input_data)

        return VideoFixtureData(
            path=input_path / "input_video.mp4",
            generator=video_generator,
            frame_data=video_generator.frame_data
            if video_generator.frame_data is not None
            else [],
        )

    return _make_fixture
