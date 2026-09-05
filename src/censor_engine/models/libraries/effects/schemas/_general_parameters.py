from dataclasses import dataclass, field

import cv2

from ._constants import OBLIQUE


@dataclass(slots=True)
class _PreProcessSettings:
    greyscale: bool = False


@dataclass(slots=True)
class _PostProcessSettings:
    alpha: float = OBLIQUE
    blur: int = 0
    glow: int = 0
    fade: int = 0
    greyscale: bool = False
    inverse: bool = False
    black_and_white: bool = False


@dataclass(slots=True)
class _MetaSettings:
    # Meta Stuff
    force_png: bool = False
    linetype: int = cv2.LINE_AA


@dataclass(slots=True)
class GeneralParameters:
    pre_processing: _PreProcessSettings = field(
        default_factory=_PreProcessSettings
    )
    post_processing: _PostProcessSettings = field(
        default_factory=_PostProcessSettings
    )
    meta: _MetaSettings = field(default_factory=_MetaSettings)
