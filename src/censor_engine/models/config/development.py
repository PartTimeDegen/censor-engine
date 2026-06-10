from pydantic import BaseModel

from censor_engine.core_engine.tools.debugger import DebugLevels


class DevelopmentConfig(BaseModel):
    """
    This is the config used for Development.

    """

    debug_level: DebugLevels = DebugLevels.NONE
