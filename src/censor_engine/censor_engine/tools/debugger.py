from enum import IntEnum


class DebugLevels(IntEnum):
    """
    This function determines the levels of debug that are active (stacks):
    -   NONE:
        -   Nothing ever happens (I'm all in)
    -   BASIC:
        -   Found Parts
        -   Censor Time
    -   DETAILED:
        -   GPU and ONNX Info
        -   Times of the Functions
        -   Censor Part Info
        -   Config Info
    -   ADVANCED:
        -   Shape IDs
        -   Merge IDs
        -   Used Censors and Styles
    -   FULL:
        -   Mask Breakdowns.

    """

    NONE = 0
    VIDEO = 1
    BASIC = 2
    TIMED = 3
    DETAILED = 4
    ADVANCED = 5
    FULL = 6
