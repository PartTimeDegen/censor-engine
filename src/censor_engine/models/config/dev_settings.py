from dataclasses import dataclass

from censor_engine.censor_engine.tools.debugger import DebugLevels


@dataclass(slots=True)
class DevConfig:
    debug_level: DebugLevels = DebugLevels.NONE
