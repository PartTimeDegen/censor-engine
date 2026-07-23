from typing import ClassVar

import cv2

from censor_engine._typing import MaskImage
from censor_engine.libraries.registries import MaskRegistry

from ._mechanisms import MaskMechanisms
from .constants import FILLED
from .enums import MaskType
from .schemas import MaskContext


class Mask:
    base_mask: ClassVar[str] = "invalid_mask"
    joint_mask: ClassVar[str] = "invalid_mask"
    single_mask: ClassVar[str] = "invalid_mask"

    mask_type: ClassVar[MaskType]

    _mechanisms: MaskMechanisms

    def __init__(self):
        self._mechanisms = MaskMechanisms()

    # Core Method
    def _pre_process(
        self, mask_context: MaskContext, **kwargs: dict
    ) -> tuple[MaskContext, dict]:
        # NOTE: This feels dirty, need to fix.
        if self.mask_type != MaskType.BAR:
            return (mask_context, kwargs)

        mask_context.mask = self.generate_joint_mask(mask_context)

        return (mask_context, kwargs)

    def generate_mask(self, mask_context: MaskContext, **kwargs) -> MaskImage:  # noqa: ANN003
        msg = f"{self.__class__.__name__} does not implement generate_mask()"
        raise NotImplementedError(msg)

    # Internal Method to Apply Different Common Settings
    def _internal_generate_mask(
        self,
        mask_context: MaskContext,
        **kwargs: dict,
    ) -> MaskImage:
        # Handle Pre-processing
        # NOTE: This is used to handle stuff like Bars, where they need a
        #       Joint Mask to work.
        new_mask_context, new_kwargs = self._pre_process(
            mask_context, **kwargs
        )
        mask_core = self.generate_mask(new_mask_context, **new_kwargs)

        if mask_context.settings.thickness == FILLED:
            return mask_core

        return self._mechanisms.hollow_mechanism(
            new_mask_context,
            mask_core,
            self.__class__.__name__,
            self.generate_mask,  # type: ignore
            **new_kwargs,
        )

    # API Helpers
    def generate_single_mask(self, mask_context: MaskContext) -> MaskImage:
        obj_mask = MaskRegistry.get(self.single_mask)
        return obj_mask().generate_mask(mask_context=mask_context)

    def generate_joint_mask(self, mask_context: MaskContext) -> MaskImage:
        obj_mask = MaskRegistry.get(self.joint_mask)
        return obj_mask().generate_mask(mask_context=mask_context)

    def get_contours(
        self,
        mask_context: MaskContext,
        *,
        mode: int = cv2.RETR_TREE,
        method: int = cv2.CHAIN_APPROX_SIMPLE,
    ):  # NOTE: Figure out type
        cont_rect, _ = cv2.findContours(
            image=mask_context.mask,
            mode=mode,
            method=method,
        )
        return cont_rect
