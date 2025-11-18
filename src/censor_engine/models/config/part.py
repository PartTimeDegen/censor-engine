from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from censor_engine.models.enums import PartState
from censor_engine.models.structs.censors import Censor


class PartSettingsConfig(BaseModel):
    """
    This is the config for the parts found in CensorEngine.

    # TODO: use_global_area

    """

    model_config = {"validate_assignment": True}

    # Meta
    name: str | None = None  # Injected from YAML key.

    # Settings
    minimum_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum score for parts to be considered valid, "
            "higher means more confident but more likely to miss parts. "
            "PTD's opinion: I keep this off since most the time it's "
            "reasonable."
        ),
    )
    censors: list[Censor] = Field(
        default_factory=list,
        description="List of the censors and their arguments used.",
    )
    shape: str = Field(
        default="box",
        description="Shape used for the part.",
        examples=[
            "box",
            "ellipse",
            "circle",
            "joint_box",
            "bar",
        ],  # TODO: Full set
    )
    margin: int | float | dict[str, float] = Field(
        default=0.0,
        description=(
            "Margin percentage for parts. Used to make the parts smaller or "
            "bigger relative to the size of the original. base is 0.0. "
            "If a dictionary is specified then width and height can be "
            "determined individually"
        ),
        ge=-1.0,
        examples=[
            1.0,
            0.6,
            1.5,
            -0.4,
            {"width": 2.0},
            {"width": 2.0, "height": 1.4},
        ],
    )

    state: PartState = Field(
        default=PartState.UNPROTECTED,
        description=(
            "Part's protection state. Used to forced overwrites for how parts "
            "are handled. This is how parts are 'revealed' (i.e., uncovered) "
            "or 'protected' (i.e., always covered)."
        ),
        examples=["unprotected", "revealed", "protected"],
    )

    protected_shape: str | None = Field(
        default=None,
        description=(
            "Shape used to protect the part, default is the part's shape"
        ),
        examples=["box", "ellipse", "circle"],
    )
    fade_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Part fade percentage",
    )

    video_part_search_region: float = Field(
        default=0.2,
        ge=0.0,
        description=(
            "Percentage of the original shape that "
            "the next frame will search for points. "
            "Used for video to maintain censors when part persistence "
            "(holding after existence)."
        ),
    )

    # Semi Meta Settings
    use_global_area: bool = Field(
        default=True,
        description="TODO: Use the entire image rather than ROI in censors.",
    )

    # --- Validators ---

    @field_validator("censors", mode="before")
    def normalise_censors(cls, censor_names: Any) -> list[Censor]:  # noqa: ANN401, N805 # Needed for PyDantic
        """
        This is used to normalise the censors to the Censor object.

        Censors can contain either the censor name, a list of censors, or a
        dictionary of censors. If it's a dictionary then it can alter the
        arguments completely however the name or a list of names will just use
        the defaults.

        :raises TypeError: Bad Censor type
        :return list[Censor]: Formatted Censor list
        """
        if censor_names is None:
            return []

        if isinstance(censor_names, str):
            return [Censor(censor_names)]

        if isinstance(censor_names, list):
            processed: list[Censor] = []
            for item in censor_names:
                match item:
                    case Censor():
                        processed.append(item)
                    case dict():
                        processed.append(Censor(**item))
                    case str():
                        processed.append(Censor(item))
                    case _:
                        msg = f"Invalid censor input: {type(item)}"
                        raise TypeError(msg)

            return processed

        msg = f"Invalid censor input: {type(censor_names)}"
        raise TypeError(msg)

    @field_validator("state", mode="before")
    def normalise_state(cls, state: Any) -> PartState:  # noqa: ANN401, N805
        """
        This normalises the state to use the PartState Enum.

        :param Any state: State
        :raises ValueError: Invalid State
        :return PartState: Formatted PartState
        """
        # If it's already a PartState, return as-is
        if isinstance(state, PartState):
            return state

        # If it's a string like "revealed", "unprotected", etc.
        if isinstance(state, str):
            try:
                return PartState[state.upper()]
            except KeyError:
                msg = f"Invalid PartState value: {state!r}"
                raise ValueError(msg)  # noqa: B904

        # If it's an integer (enum numeric value)
        if isinstance(state, int):
            try:
                return PartState(state)
            except ValueError:
                msg = f"Invalid PartState numeric value: {state!r}"
                raise ValueError(msg)  # noqa: B904

        msg = f"Invalid PartState type: {type(state)}"
        raise ValueError(msg)

    @field_validator("margin", mode="before")
    def validate_margin(cls, margin_data: Any) -> int | float | dict:  # noqa: ANN401, N805
        """
        This is used to validate the margin setting.

        # TODO: I could probably migrate the *old* system from the core code
        #       to here, so that code is simpler and the formatting isn't
        #       somewhere in God knows where.

        # TODO: Need to write block that handles dict values being below zero.

        :param Any margin_data: Margin Config
        :raises TypeError: Invalid type
        :raises ValueError: Margin below zero
        :return int | float | dict: Margin data
        """
        if not isinstance(margin_data, int | float | dict):
            msg = f"Invalid type for margin: {type(margin_data)}"
            raise TypeError(msg)

        if not isinstance(margin_data, dict) and float(margin_data < -1.0):
            msg = f"Margin size cannot be below -1.0: {margin_data}"
            raise ValueError(msg)

        return margin_data


class MergingConfig(BaseModel):
    """
    This holds the merge information. The merge information is used for the
    clustering part of the code.

    # TODO: MERGE THIS WITH MERGE GROUPS

    # TODO: merge_range

    """

    merge_range: float = Field(
        default=-1.0,
        description=(
            "NOT IMPLEMENTED: "
            "Range relative to part size that the part will search for merging"
        ),
    )
    merge_groups: list[list[str]] = Field(
        default_factory=list,
        description="Groups for parts to merge",
        examples=[
            [
                ["FEMALE_BREAST_EXPOSED", "FEMALE_BREAST_COVERED"],
                ["FEMALE_GENITALIA_EXPOSED", "FEMALE_GENITALIA_COVERED"],
            ],
            [
                ["FEMALE_BREAST_EXPOSED", "FEMALE_BREAST_COVERED"],
            ],
        ],
    )

    @field_validator("merge_groups")
    def validate_merge_groups(cls, merge_groups: Any) -> list[list[str]]:  # noqa: ANN401, N805
        """
        Validates the correct usage of merge groups.

        # TODO: Implement fixing behaviour instead of raising an error

        :param Any merge_groups: Merge Groups
        :raises TypeError: Didn't use a list of lists
        :return list[list[str]]: Output
        """
        if merge_groups and not isinstance(merge_groups[0], list):
            msg = "merge_groups must be a list of lists"
            raise TypeError(msg)
        return merge_groups


class PartInformationConfig(BaseModel):
    """
    This is used to hold all the parts settings and manage them, i.e., this
    holds the settings for all parts.

    """

    enabled_parts: list[str] = Field(
        default_factory=list,
        description="List of enabled parts",
        examples=[
            "all",  # TODO: Account for this, it's in the code just not here
            ["FEMALE_BREAST_EXPOSED", "FEMALE_BREAST_COVERED"],
            [
                "FEMALE_BREAST_EXPOSED",
                "FEMALE_BREAST_COVERED",
                "FEMALE_GENITALIA_EXPOSED",
                "FEMALE_GENITALIA_COVERED",
            ],
        ],
    )
    parts_settings: dict[str, PartSettingsConfig] = Field(
        default_factory=dict, description="Settings per part"
    )
    merge_settings: MergingConfig = Field(
        default_factory=MergingConfig, description="Merge configuration"
    )

    @model_validator(mode="after")  # type: ignore
    def sync_part_names_with_keys(cls, model: "PartInformationConfig"):  # noqa: ANN201, N805
        """
        Updates the parts to have their key in the config, as their name
        attribute.
        """
        for key, part in model.parts_settings.items():
            part.name = key
        return model
