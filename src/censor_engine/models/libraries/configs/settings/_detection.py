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

# NOTE: I just pulled these from a printout, turns out a property needs an
#       instance to be called so I couldn't automate it.
DEFAULT_ENABLED_PARTS = (
    "NUDENET_FACE_FEMALE",
    "NUDENET_ARMPITS_EXPOSED",
    "NUDENET_ARMPITS_COVERED",
    "NUDENET_FEMALE_BREAST_EXPOSED",
    "NUDENET_FEMALE_BREAST_COVERED",
    "NUDENET_BELLY_EXPOSED",
    "NUDENET_BELLY_COVERED",
    "NUDENET_BUTTOCKS_EXPOSED",
    "NUDENET_BUTTOCKS_COVERED",
    "NUDENET_ANUS_EXPOSED",
    "NUDENET_ANUS_COVERED",
    "NUDENET_FEMALE_GENITALIA_EXPOSED",
    "NUDENET_FEMALE_GENITALIA_COVERED",
    "NUDENET_FEET_EXPOSED",
    "NUDENET_FEET_COVERED",
    "NUDENET_FACE_MALE",
    "NUDENET_MALE_GENITALIA_EXPOSED",
    "NUDENET_MALE_BREAST_EXPOSED",
)


class PartSettings(BaseModel):
    """
    _summary_.

    Notes:
        -   It might be worth to make masks a model that generates from the
            register.

    """

    mask: str = DEFAULT_MASK

    minimum_score: BoundPercentage = 0.0
    state: PartState = PartState.UNPROTECTED
    fade: BoundPercentage = 0.0

    use_global_area: bool = True

    censors: ListOfCensors = Field(default_factory=list)
    margins: Margins = Field(default_factory=Margins)
    tracking_margin: Margins = Field(default_factory=Margins)

    _normalise_censor = field_validator("censors", mode="before")(
        normalise_censors
    )
    _normalise_margins = field_validator(
        "margins", "tracking_margin", mode="before"
    )(normalise_margins)

    _convert_sate = field_validator("state", mode="before")(convert_state)

    @field_validator("mask", mode="before")
    @classmethod
    def convert_mask_to_upper(cls, value: str) -> str:
        return value.upper()


class DetectionSettings(BaseModel):
    enabled_parts: list[str] | None = None
    default_settings: PartSettings = Field(default_factory=PartSettings)
    part_settings: dict[str, PartSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def apply_enabled_parts(self):
        if self.enabled_parts is not None:
            return self

        self.enabled_parts = list(DEFAULT_ENABLED_PARTS)
        return self

    @model_validator(mode="after")
    def apply_defaults(self):
        resolved_parts: dict[str, PartSettings] = {}
        if self.enabled_parts is None:
            enabled_parts = list(DEFAULT_ENABLED_PARTS)
        else:
            enabled_parts = self.enabled_parts

        for part in enabled_parts:
            # Check if Part was Mentioned
            custom_part_config = self.part_settings.get(part)

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

            # Handle Revealed State Having no Censor by Default
            # NOTE: If this isn't done then the revealed part just inherits the
            #       default censors which re-censors it.
            if (
                custom_part_config.state == PartState.REVEALED
                and custom_part_config.censors == []
            ):
                resolved_parts[part].censors = []

        self.part_settings = resolved_parts
        return self

    @model_validator(mode="after")
    def convert_enabled_parts_to_upper_case(self):
        if self.enabled_parts is None:
            msg = "Pydantic is wrong"
            raise TypeError(msg)
        self.enabled_parts = [part.upper() for part in self.enabled_parts]
        return self
