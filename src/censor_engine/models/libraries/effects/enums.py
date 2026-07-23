from enum import IntEnum


class EffectType(IntEnum):
    INVALID = 0
    BLUR = 1
    OVERLAY = 2
    COLOUR = 3
    DEV = 4
    EDGE_DETECTION = 5
    NOISE = 6
    PIXELATION = 7
    STYLISATION = 8
    TEXT = 9
    TRANSPARENCY = 10
