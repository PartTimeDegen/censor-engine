from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gdown  # type: ignore
import pydload  # type: ignore

# import tensorflow as tf  # type: ignore # NOTE: I can't run this due to my shitty Xeon CPU
from censor_engine.typing import Image


@dataclass(slots=True)
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
    part_id: int = 0

    def set_part_id(self, number: int) -> None:
        self.part_id = number


class AIModel:
    """
    Handles downloading and loading AI models separately from code packages.
    """

    ai_model_folder_name: str = "~/.ai_models"
    model_path: Path | None = None
    model: Any = None

    def download_model(self, url: str):
        home = Path.home()
        model_folder = home / self.ai_model_folder_name.lstrip("~").lstrip("/")
        model_folder.mkdir(parents=True, exist_ok=True)

        model_path = model_folder / Path(url).name

        if not model_path.exists():
            print(f"Downloading the checkpoint to {model_path}")
            pydload.dload(url, save_to_path=str(model_path), max_time=None)  # type: ignore

        self.model_path = model_path

    def download_google_drive_model(self, url: str):
        home = Path.home()
        model_folder = home / self.ai_model_folder_name.lstrip("~").lstrip("/")
        model_folder.mkdir(parents=True, exist_ok=True)

        model_path = model_folder / Path(url).name

        if not model_path.exists():
            print(f"Downloading the checkpoint to {model_path}")
            gdown.download(url, str(model_path), max_time=None)  # type: ignore

        self.model_path = model_path

    def load_model(self):
        if self.model_path is None:
            raise ValueError(
                "Model path is not set. Please download a model first."
            )

        # ext = self.model_path.suffix.lower().lstrip(".")

        # # PyTorch model loading
        # if ext == "pt":
        #     self.model = torch.load(str(self.model_path))
        #     self.model.eval()
        # else:
        #     raise ValueError(f"Unsupported model format: {ext}")

    def predict(self, input_data: Any):
        if self.model is None:
            raise ValueError("Missing AI model. Please load the model first.")

        # if isinstance(self.model, torch.nn.Module):
        #     with torch.no_grad():
        #         input_tensor = torch.tensor(input_data, dtype=torch.float32)
        #         output = self.model(input_tensor)
        #         return output.numpy()
        # else:
        #     raise ValueError("Cannot find supported model type")

    def proceed_model(self, url: str, input_data: Any):
        if "drive.google" in url:
            self.download_google_drive_model(url)
        else:
            self.download_model(url)
        self.load_model()
        return self.predict(input_data)


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
    def detect_image(
        self, file_images_or_path: str
    ) -> list[DetectedPartSchema]:
        raise NotImplementedError

    @abstractmethod
    def detect_batch(
        self, file_images_or_paths: list[str] | list[Image], batch_size: int
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
