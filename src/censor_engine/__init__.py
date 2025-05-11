import os
from censor_engine.censor_engine import CensorEngine
import censor_engine

__all__ = ["CensorEngine"]


APPROVED_FORMATS_IMAGE = [".jpg", ".jpeg", ".png", ".webp"]
APPROVED_FORMATS_VIDEO = [".mp4", ".webm"]

PROJECT_ROOT = os.sep.join((censor_engine.__file__).split(os.sep)[:-1])

CONFIGS_FOLDER = os.path.join(PROJECT_ROOT, "configs")
