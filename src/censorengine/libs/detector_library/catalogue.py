from censorengine.libs.detector_library.detectors.multi_detectors import (
    NudeNetDetector,
)
from censorengine.libs.detector_library.detectors.determination_tools import (
    ImageGenreDeterminer,
)

"""
This is used for enabling new models. You may notice it's different from the
other catalogue files, that's due to the fact it doesn't need to use config 
files (yet, maybe, might be overkill)
"""


enabled_detectors = [
    NudeNetDetector(),
]

enabled_determiners = [
    ImageGenreDeterminer(),
]
