from enum import IntEnum


class PartState(IntEnum):
    UNPROTECTED = 1
    REVEALED = 2
    PROTECTED = 3


class ShapeType(IntEnum):
    BASIC = 1
    JOINT = 2
    BAR = 3
