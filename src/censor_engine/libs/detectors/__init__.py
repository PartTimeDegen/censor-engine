from censor_engine.models.lib_models.detectors import Detector

from .determination_tools import ImageGenreDeterminer
from .multi_detectors import NudeNetDetector

"""
This is used for enabling new models. You may notice it's different from the
other catalogue files, that's due to the fact it doesn't need to use config 
files (yet, maybe, might be overkill)
"""


enabled_detectors: list[Detector] = [
    NudeNetDetector(),  # type: ignore # FIXME at some point
]

enabled_determiners = [
    ImageGenreDeterminer(),
]
