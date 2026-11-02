from enum import IntEnum, StrEnum


class ModelType(IntEnum):
    TORCH = 1
    SAM = 2
    MEDIAPIPE = 3


class OutputType(StrEnum):
    PART = "part"
    DEPTH = "depth"
