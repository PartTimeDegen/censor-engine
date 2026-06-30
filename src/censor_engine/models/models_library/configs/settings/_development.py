from pydantic import BaseModel, field_validator

from censor_engine.models.models_core.tools.debugger.enums import DebugLevels
from censor_engine.models.models_library.configs._validators import (
    convert_debug_level,
)


class DevelopmentSettings(BaseModel):
    """
    This is the config used for Development.

    """

    debug_level: DebugLevels = DebugLevels.NONE

    _convert_merge_method = field_validator("debug_level", mode="before")(
        convert_debug_level
    )
