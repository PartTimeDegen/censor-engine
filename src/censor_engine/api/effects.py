from dataclasses import dataclass


@dataclass(slots=True)
class EffectContext:
    placeholder: str
