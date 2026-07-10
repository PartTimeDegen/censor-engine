from pydantic import BaseModel, Field, field_validator, model_validator

from censor_engine.models.enums import PartState
from censor_engine.models.libraries.configs._constants import DEFAULT_MASK
from censor_engine.models.libraries.configs._helper_types import (
    BoundPercentage,
    ListOfCensors,
)
from censor_engine.models.libraries.configs._validators import (
    convert_state,
    normalise_censors,
    normalise_margins,
)
from censor_engine.models.libraries.configs.settings.schemas import Margins


class PartSettings(BaseModel):
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
    margins: Margins = Field(default_factory=Margins)
    tracking_margin: Margins = Field(default_factory=Margins)

    _normalise_censor = field_validator("censors", mode="before")(
        normalise_censors
    )
    _normalise_margins = field_validator(
        "margins", "tracking_margin", mode="before"
    )(normalise_margins)

    _convert_sate = field_validator("state", mode="before")(convert_state)


class DetectionSettings(BaseModel):
    enabled_parts: list[str] = Field(default_factory=list)
    default_settings: PartSettings = Field(default_factory=PartSettings)
    parts: dict[str, PartSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def apply_defaults(self):
        resolved_parts = {}

        for part in self.enabled_parts:
            # Check if Part was Mentioned
            custom_part_config = self.parts.get(part)

            # Use Default if Not
            if custom_part_config is None:
                resolved_parts[part] = self.default_settings.model_copy(
                    deep=True
                )
                continue

            # Update Defaults with Found settings
            resolved_parts[part] = PartSettings.model_validate(
                {
                    **self.default_settings.model_dump(),
                    **custom_part_config.model_dump(exclude_unset=True),
                }
            )

        self.parts = resolved_parts
        return self
