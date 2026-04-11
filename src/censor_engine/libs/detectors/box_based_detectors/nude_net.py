from importlib import resources

import onnxruntime as ort  # type: ignore
from nudenet import NudeDetector  # type: ignore

from censor_engine.models.lib_models.detectors import (
    DetectedPartSchema,
    Detector,
)
from censor_engine.typing import Image


class GpuNudeDetector(NudeDetector):
    """
    This detector is to enable GPU support for NudeNet, for some reason the dev
    commented out support and forgot to uncomment it.

    """

    def __init__(self, *, use_gpu: bool = True):
        # Get the NudeNet Model from Package
        # NOTE: For some reason he put the entire model in here, oh well it's
        #       easier for me.
        model_path = resources.files("nudenet") / "320n.onnx"

        # Check that a GPU Provider is Possible
        providers = ["CPUExecutionProvider"]

        if (
            use_gpu
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Create Session
        self.onnx_session = ort.InferenceSession(
            str(model_path), providers=providers
        )
        print(f"Using GPU for NudeNet: {self.is_gpu()}")

        # Setup Model as per Nude Net
        model_inputs = self.onnx_session.get_inputs()

        self.input_width = 320
        self.input_height = 320
        self.input_name = model_inputs[0].name

    def is_gpu(self):
        return "CUDAExecutionProvider" in self.onnx_session.get_providers()


class NudeNetDetector(Detector):
    """
    This Detector is the code of CensorEngine, it is the NudeNet model.

    This handles the core labels of the engine.

    """

    model_name: str = "NudeNet"
    model_classifiers: tuple[str, ...] = (
        "FACE_FEMALE",
        "ARMPITS_EXPOSED",
        "ARMPITS_COVERED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "BELLY_EXPOSED",
        "BELLY_COVERED",
        "BUTTOCKS_EXPOSED",
        "BUTTOCKS_COVERED",
        "ANUS_EXPOSED",
        "ANUS_COVERED",
        "FEMALE_GENITALIA_EXPOSED",
        "FEMALE_GENITALIA_COVERED",
        "FEET_EXPOSED",
        "FEET_COVERED",
        "FACE_MALE",
        "MALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
    )
    model_object = GpuNudeDetector()

    def detect_image(
        self,
        file_images_or_path: str,
    ) -> list[DetectedPartSchema]:
        return [
            DetectedPartSchema(
                label=found_part["class"],
                score=found_part["score"],
                relative_box=found_part["box"],
            )
            for _, found_part in enumerate(
                self.model_object.detect(file_images_or_path),
            )
        ]

    def detect_batch(
        self,
        file_images_or_paths: list[str] | list[Image],
        batch_size: int,
    ) -> dict[int, list[DetectedPartSchema]]:
        output = self.model_object.detect_batch(
            file_images_or_paths,
            batch_size,
        )

        dict_output = {}
        for index, image in enumerate(output):
            dict_output[index] = [
                DetectedPartSchema(
                    part_id=index,
                    label=found_part["class"],
                    score=found_part["score"],
                    relative_box=found_part["box"],
                )
                for index, found_part in enumerate(image)
            ]

        return dict_output
