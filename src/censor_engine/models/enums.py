from enum import IntEnum


class PartState(IntEnum):
    UNPROTECTED = 1
    REVEALED = 2
    PROTECTED = 3


class ShapeType(IntEnum):
    BASIC = 1
    JOINT = 2
    BAR = 3


class StyleType(IntEnum):
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
