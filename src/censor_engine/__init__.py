import os

import censor_engine
from censor_engine.censor_engine import CensorEngine

__all__ = ["CensorEngine"]


PROJECT_ROOT = os.sep.join((censor_engine.__file__).split(os.sep)[:-1])
