from enum import IntEnum


class PartState(IntEnum):
    UNPROTECTED = 1
    REVEALED = 2
    PROTECTED = 3


class MergeMethod(IntEnum):
    NONE = 1
    GROUPS = 2
    PARTS = 3
    FULL = 4
    ALL = 5
