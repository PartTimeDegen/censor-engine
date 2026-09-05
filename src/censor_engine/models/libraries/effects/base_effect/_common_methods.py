import cv2

from censor_engine._typing import Image
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)


class CommonProcessingMethods:
    def _create_greyscale(
        self, effect_context: EffectContext, **kwargs: dict
    ) -> Image:
        mask_image = cv2.cvtColor(effect_context.image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)  # type: ignore
