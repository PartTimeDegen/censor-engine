from pydantic import BaseModel, Field, field_validator

from censor_engine.models.enums import MergeMethod
from censor_engine.models.libraries.configs._helper_types import (
    ListOfCensors,
    MarginPercentage,
)
from censor_engine.models.libraries.configs._validators import (
    convert_merge_method,
    normalise_censors,
)


class _Merging(BaseModel):
    method: MergeMethod = Field(default=MergeMethod.NONE)
    merge_range: MarginPercentage = 0.0

    _convert_merge_method = field_validator("method", mode="before")(
        convert_merge_method
    )


class ImageSettings(BaseModel):
    merging: _Merging = Field(default_factory=_Merging)
    reverse_censor: ListOfCensors = Field(default_factory=list)

    _normalise_censor = field_validator("reverse_censor", mode="before")(
        normalise_censors
    )
