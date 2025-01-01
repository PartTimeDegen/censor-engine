from abc import abstractmethod
from dataclasses import dataclass


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
    relative_box: tuple[int, int, int, int]


class Detector:
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
    def detect_image(self, file_path: str) -> list[DetectedPartSchema]:
        raise NotImplementedError
