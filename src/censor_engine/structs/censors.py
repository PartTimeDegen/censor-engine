from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Censor:
    """
    This is used to make the handling of censors (effects) a bit more easy to
    process.

    """

    effect: str
    parameters: dict[str, Any] = field(default_factory=dict)
