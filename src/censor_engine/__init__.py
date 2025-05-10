import os
from censor_engine.backend.models.censor_engine.censor_engine.censor_engine import (
    CensorEngine,
)

__all__ = ["CensorEngine"]


# TODO: Fixme
import os
import censorengine


APPROVED_FORMATS_IMAGE = [".jpg", ".jpeg", ".png", ".webp"]
APPROVED_FORMATS_VIDEO = [".mp4", ".webm"]

PROJECT_ROOT = os.sep.join((censorengine.__file__).split(os.sep)[:-1])

CONFIGS_FOLDER = os.path.join(PROJECT_ROOT, "configs")
