from censor_engine.censor_engine.tools.debugger import DebugLevels
from dataclasses import dataclass


@dataclass(slots=True)
class DevConfig:
    debug_level: DebugLevels = DebugLevels.NONE
