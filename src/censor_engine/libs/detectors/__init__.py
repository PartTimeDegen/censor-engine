from censor_engine.lib_models.detectors import Detector
from censor_engine.libs.detector_library.detectors.multi_detectors import (
    NudeNetDetector,
)
from censor_engine.libs.detector_library.detectors.determination_tools import (
    ImageGenreDeterminer,
)

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
