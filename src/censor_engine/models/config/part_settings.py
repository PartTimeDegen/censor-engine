from dataclasses import dataclass, field

from censor_engine.models.enums import PartState
from censor_engine.models.structs.censors import Censor


@dataclass(slots=True)
class PartSettingsConfig:
    # Meta
    name: str = "MISSING_NAME"

    # Settings
    minimum_score: float | None = 0.20
    censors: list[Censor] = field(default_factory=list)
    shape: str = "box"
    margin: int | float | dict[str, float] = 0.0
    state: PartState = PartState.UNPROTECTED
    protected_shape: str | None = None
    fade_percent: int = 0  # 0 - 100
    video_part_search_region: float = 0.2  # bigger than 0.0

    # Semi Meta Settings
    use_global_area: bool = True

    def __str__(self):
        return self.name

    def __post_init__(self):
        """
        This section is to handle that the incoming data are Python builtins,
        not for example Censor or PartState.

        TODO: This doesn't confirm information is the right type, use pydantic
        at some point.

        """
        # Censors
        self.censors = [
            Censor(**censor) if isinstance(censor, dict) else censor
            for censor in self.censors
        ]

        # Part State
        if isinstance(self.state, str):
            self.state = getattr(PartState, self.state.upper())
            if not self.state:
                raise ValueError(f"Invalid PartState value: {self.state}")

        # Margin
        if not isinstance(self.margin, int | float | dict):
            raise TypeError(f"Invalid type for margin: {type(self.margin)}")


@dataclass(slots=True)
class MergingConfig:
    merge_range: float | int = -1.0  # TODO
    merge_groups: list[list[str]] = field(default_factory=list)

    def __post_init__(self):
        # Type Narrow to Float
        if isinstance(self.merge_range, int):
            self.merge_range = float(self.merge_range)


@dataclass(slots=True)
class PartInformationConfig:
    enabled_parts: list[str] = field(default_factory=list)

    parts_settings: dict[str, PartSettingsConfig] = field(default_factory=dict)
    merge_settings: MergingConfig = field(default_factory=MergingConfig)
