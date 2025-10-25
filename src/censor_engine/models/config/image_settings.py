from dataclasses import dataclass, field

from censor_engine.models.enums import MergeMethod
from censor_engine.models.structs.censors import Censor


@dataclass(slots=True)
class RenderingConfig:
    batch_size: int = 4  # TODO: Multi-threading
    merge_method: MergeMethod = MergeMethod.GROUPS

    def __post_init__(self):
        # Part State
        if isinstance(self.merge_method, str):
            self.merge_method = getattr(MergeMethod, self.merge_method.upper())
            if not self.merge_method:
                msg = f"Invalid MergeMethod value: {self.merge_method}"
                raise ValueError(
                    msg,
                )


@dataclass(slots=True)
class AIConfig:
    ai_model_downscale_factor: int = 1  # TODO: Implement


@dataclass(slots=True)
class ReverseCensorConfig:
    censors: list[Censor] = field(default_factory=list)

    def __post_init__(self):
        """Ensure censors are converted to Censor objects."""
        self.censors = [
            Censor(**censor) if isinstance(censor, dict) else censor
            for censor in self.censors
        ]
