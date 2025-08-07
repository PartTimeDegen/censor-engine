from dataclasses import dataclass, field
from typing import Literal

from censor_engine.models.structs.censors import Censor


@dataclass(slots=True)
class RenderingConfig:
    batch_size: int = 4  # TODO
    merge_method: Literal[
        "none",
        "groups",
        "parts",
        "full",
    ] = "full"


@dataclass(slots=True)
class AIConfig:
    ai_model_downscale_factor: int = 1  # TODO


@dataclass(slots=True)
class ReverseCensorConfig:
    censors: list[Censor] = field(default_factory=list)

    def __post_init__(self):
        """
        Ensure censors are converted to Censor objects
        """
        self.censors = [
            Censor(**censor) if isinstance(censor, dict) else censor
            for censor in self.censors
        ]
