from pydantic import BaseModel, Field, NonNegativeFloat

from censor_engine.models.libraries.configs._constants import (
    DEFAULT_SIZE_DIFF_PERCENTAGE,
)
from censor_engine.models.libraries.configs._helper_types import (
    Frames,
    Seconds,
)


class _FPS(BaseModel):
    censoring: Frames = 1
    output: Frames = 1


class _Stability(BaseModel):
    size_difference_percentage: NonNegativeFloat = DEFAULT_SIZE_DIFF_PERCENTAGE


class _Persistance(BaseModel):
    censor_hold: Seconds = 0.0


class VideoSettings(BaseModel):
    fps: _FPS = Field(default_factory=_FPS)
    stability: _Stability = Field(default_factory=_Stability)
    persistance: _Persistance = Field(default_factory=_Persistance)
