from enum import IntEnum, auto


class EffectType(IntEnum):
    INVALID = auto()
    BLUR = auto()
    OVERLAY = auto()
    COLOUR = auto()
    DEV = auto()
    EDGE_DETECTION = auto()
    NOISE = auto()
    PIXELATION = auto()
    CRYSTALLISATION = auto()
    POLYGON = auto()
    STYLISATION = auto()
    TEXT = auto()
    TRANSPARENCY = auto()
