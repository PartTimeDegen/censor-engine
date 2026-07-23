from collections.abc import Callable
from copy import deepcopy

import cv2

from censor_engine._typing import MaskImage
from censor_engine.models.libraries.configs.settings.schemas import Margins

from .schemas import MaskContext


class MaskMechanisms:
    _symmetrical_shapes: tuple[str] = ("circle",)

    def hollow_mechanism(
        self,
        mask_context: MaskContext,
        mask_core: MaskImage,
        _model_name: str,
        _generate_function: Callable[[MaskContext, dict], MaskImage],
        **kwargs,
    ) -> MaskImage:
        # Create Copy
        new_mask_context = deepcopy(mask_context)

        # Make Margins
        percentage_gone = 1 - new_mask_context.settings.thickness
        bbox = new_mask_context.part_properties.bbox
        if bbox is None:
            raise TypeError

        # Find Smallest Size
        height = bbox.height
        width = bbox.width
        min_dim = min(height, width)

        # Calculate New Margins
        margin_px = int(min_dim * percentage_gone)
        margins = Margins(height=-margin_px / height, width=-margin_px / width)

        # Handle Cases like Circle where it will always be Symmetrical
        if _model_name.lower() in self._symmetrical_shapes:
            margins = Margins(height=-percentage_gone, width=-percentage_gone)

        # Rescale BBox
        new_bbox = bbox.rescale_bbox(margins)
        new_mask_context.part_properties.bbox = new_bbox

        # Get the New Mask
        mask_subtraction = _generate_function(new_mask_context, **kwargs)  # type: ignore
        return cv2.subtract(mask_core, mask_subtraction)  # type: ignore
