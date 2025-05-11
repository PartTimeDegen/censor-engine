from abc import abstractmethod
from dataclasses import dataclass
import os
from typing import Any

import pydload  # type: ignore
import gdown  # type: ignore

# import tensorflow as tf  # type: ignore # NOTE: I can't run this due to my shitty Xeon CPU
import torch

from censor_engine.typing import CVImage


@dataclass
class DetectedPartSchema:
    """
    This is the schema expected to be returned by core and customer detectors.

    Additional params might be added over time due to additional features. This
    model also allows an abstraction layer for models, since the same features
    might not share the same names.

    E.g., When I was trying to find better /more up to date models, I found
    another version of NudeNet someone used, however they changed the "class"
    key to "label" (probably same reason I did, Python has a `class` keyword
    which is annoying), so I had to change it or adapt it (forgot which) to fix
    it.

    """

    label: str
    score: float
    relative_box: tuple[int, int, int, int]  # X, Y, Width, Height


class AIModel:
    """
    It seems that the standard for AI packages for Python is to give you the
    code via Pip but then give you the model separately (probably to alleviate
    the payload).

    This class is to give the AI using models the ability to download and mount
    the models.

    Downloading is simple enough since it's just getting the link.

    Mounting requires more work especially since I'm not an AI dev (yet lol)

    """

    ai_model_folder_name: str = "~/.ai_models"
    model_path: str
    ai_model: Any

    def download_model(self, url: str):
        # Download the Model When the Package Doesn't -.-
        home = os.path.expanduser("~")
        model_folder = os.path.join(home, self.ai_model_folder_name, "")
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        model_path = os.path.join(model_folder, os.path.basename(url))

        if not os.path.exists(model_path):
            print("Downloading the checkpoint to", model_path)
            pydload.dload(url, save_to_path=model_path, max_time=None)  # type: ignore

        self.model_path = model_path

    def download_google_drive_model(self, url: str):
        # Download the Model When the Package Doesn't -.-
        home = os.path.expanduser("~")
        model_folder = os.path.join(home, self.ai_model_folder_name, "")
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        model_path = os.path.join(model_folder, os.path.basename(url))

        if not os.path.exists(model_path):
            print("Downloading the checkpoint to", model_path)
            gdown.download(url, model_path, max_time=None)  # type: ignore

        self.model_path = model_path

    def load_model(self):
        file_extension = self.model_path.split(".")[-1]

        # # TensorFlow Models
        # if file_extension in ["h5", "savedmodel"]:
        #     self.model = tf.keras.models.load_model(self.model_path)

        # PyTorch
        if file_extension == "pt":
            self.model = torch.load(self.model_path)
            self.model.eval()

        else:
            raise ValueError(f"Unsupported model format: {file_extension}")

    def predict(self, input_data: Any):
        if self.model is None:
            raise ValueError("Missing AI model.")

        # # TensorFlow Model
        # if isinstance(self.model, tf.keras.Model):
        #     return self.model.predict(input_data)

        # PyTorch
        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
                return self.model(input_tensor).numpy()

        else:
            raise ValueError("Cannot find model type")

    def proceed_model(self, url: str, input_data: Any):
        if "drive.google" in url:
            self.download_google_drive_model(url)
        else:
            self.download_model(url)
        self.load_model()
        self.predict(input_data)


class Detector(AIModel):
    """
    This is the model used for detectors, it's pretty simple, just maintains a
    valid method to use/overwrite, and some meta information.

    The meta information isn't useful for code (custom models may vary),
    however for a documentation POV, it's useful to know the internal name
    (maybe logging) and what it's producing (Trust me, it's better than having
    to find NudeNet's repo to find the classifiers).

    :raises NotImplementedError: This is just to throw an error to ensure the model devs know they didn't properly implement the method under the correct name
    """

    model_name: str
    model_classifiers: tuple[str, ...]

    @abstractmethod
    def detect_image(self, file_images_or_path: str) -> list[DetectedPartSchema]:
        raise NotImplementedError

    @abstractmethod
    def detect_batch(
        self, file_images_or_paths: list[str] | list[CVImage], batch_size: int
    ) -> dict[int, list[DetectedPartSchema]]:
        raise NotImplementedError


class Determiner(AIModel):
    """
    This class is a variant of Detector used from "determiners", it uses AI
    however rather than returning objects like the Detectors do, it determines
    (shocking I know), this is mostly used for tools where refinement can be
    made from the information returned.

    For instance this class was made for the "ImageGenreDeterminer" which
    determines if an image is hentai (well drawing) or real, such that one can
    make refinements based on this info (check the class as it explains more).
    """

    model_name: str
    model_classifiers: tuple[str, ...]

    @abstractmethod
    def determine_image(self, file_path: str) -> str:
        raise NotImplementedError
