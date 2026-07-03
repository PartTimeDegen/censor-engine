from pydantic import BaseModel, Field, field_validator, model_validator

from censor_engine.models.enums import PartState
from censor_engine.models.libraries.configs._constants import DEFAULT_MASK
from censor_engine.models.libraries.configs._helper_types import (
    BoundPercentage,
    ListOfCensors,
    MarginPercentage,
)
from censor_engine.models.libraries.configs._validators import (
    convert_state,
    normalise_censors,
    normalise_margins,
)


class _Margins(BaseModel):
    height: MarginPercentage = 0.0
    width: MarginPercentage = 0.0


class _PartSettings(BaseModel):
    """
    _summary_.

    Notes:
        -   It might be worth to make masks a model that generates from the
            register.

    """

    mask: str = DEFAULT_MASK

    minimum_score: BoundPercentage = 0.0
    state: PartState = PartState.UNPROTECTED  # TODO: Convert
    protection_mask: str | None = None
    fade: BoundPercentage = 0.0

    use_global_area: bool = True

    censors: ListOfCensors = Field(default_factory=list)  # TODO: Convert
    margins: _Margins = Field(default_factory=_Margins)
    tracking_margin: _Margins = Field(default_factory=_Margins)

    _normalise_censor = field_validator("censors", mode="before")(
        normalise_censors
    )
    _convert_sate = field_validator("state", mode="before")(convert_state)

    @field_validator("margins", "tracking_margin", mode="before")
    @classmethod
    def normalise_margins(cls, v):  # noqa: ANN001, ANN206
        processed_data = normalise_margins(v)
        return _Margins(**processed_data)


class DetectionSettings(BaseModel):
    enabled_parts: list[str] = Field(default_factory=list)
    default_settings: _PartSettings = Field(default_factory=_PartSettings)
    parts: dict[str, _PartSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def apply_defaults(self):
        resolved_parts = {}

        for part in self.parts:
            # Check if Part was Mentioned
            custom_part_config = self.parts.get(part)

            # Use Default if Not
            if custom_part_config is None:
                resolved_parts[part] = self.default_settings.model_copy(
                    deep=True
                )
                continue

            # Update Defaults with Found settings
            resolved_parts[part] = self.default_settings.model_copy(
                update=custom_part_config.model_dump(exclude_unset=True),
                deep=True,
            )

        self.parts = resolved_parts
        return self
