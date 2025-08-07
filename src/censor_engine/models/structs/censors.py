from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Censor:
    """
    This is used to make the handling of censors (styles) a bit more easy to
    process.

    """

    function: str
    args: dict[str, Any] = field(default_factory=dict)
