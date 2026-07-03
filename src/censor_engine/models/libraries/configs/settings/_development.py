from pydantic import BaseModel, field_validator

from censor_engine.models.core.tools.debugger.enums import DebugLevel
from censor_engine.models.libraries.configs._validators import (
    convert_debug_level,
)


class DevelopmentSettings(BaseModel):
    """
    This is the config used for Development.

    """

    debug_level: DebugLevel = DebugLevel.NONE

    _convert_merge_method = field_validator("debug_level", mode="before")(
        convert_debug_level
    )
