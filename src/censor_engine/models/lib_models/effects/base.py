import cv2

from censor_engine.api.effects import EffectContext
from censor_engine.constant import DIM_COLOUR, DIM_GREY, DIM_RGBA
from censor_engine.models.enums import EffectType
from censor_engine.typing import Image, ProcessedImage

from .mixin_contour_masking import MixinContourMasking
from .mixin_image_blending import MixinImageBlending


class Effect(MixinContourMasking, MixinImageBlending):
    # Information
    effect_type: EffectType = EffectType.INVALID

    # Supporting Information
    force_png: bool = False
    default_linetype: int = cv2.LINE_AA
    using_reverse_censor: bool = False

    def _merge_processed_to_input_image(
        self,
        effect_context: EffectContext,
        original_image: Image,
        processed_image: ProcessedImage,
    ) -> ProcessedImage:

        processed_image = cv2.addWeighted(
            processed_image,
            effect_context.alpha,
            original_image,
            1 - effect_context.alpha,
            0,
        )

        if effect_context.fade_width > 0:
            return self.blend_with_fade(
                original_image,
                processed_image,
                effect_context.mask,
                effect_context.fade_width,
                gradient_mode=effect_context.fade_gradient_mode,
                mask_thickness=effect_context.mask_thickness,
            )

        return self.apply_hard_mask(
            original_image,
            processed_image,
            effect_context.mask,
        )

    def internal_run_effect(
        self,
        effect_context: EffectContext,
        **kwargs,  # noqa: ANN003
    ) -> ProcessedImage:
        # Apply Core Effect
        original_image = effect_context.original_image
        processed_image = self.apply_effect(effect_context, **kwargs)

        if processed_image.shape[2] == DIM_RGBA:
            if effect_context.shape[2] == DIM_GREY:
                original_image = cv2.cvtColor(
                    effect_context.original_image,
                    cv2.COLOR_GRAY2RGBA,
                )
            elif effect_context.shape[2] == DIM_COLOUR:
                original_image = cv2.cvtColor(
                    effect_context.original_image,
                    cv2.COLOR_RGB2RGBA,
                )

        # Apply Post-processing Effects
        return self._merge_processed_to_input_image(
            effect_context,
            original_image,
            processed_image,
        )

    def apply_effect(
        self,
        effect_context: EffectContext,
        **kwargs,  # noqa: ANN003
    ) -> ProcessedImage:
        raise NotImplementedError

    def change_linetype(self, enable_aa: bool) -> None:  # noqa: FBT001
        # TODO: FBT error needs to be addressed
        if enable_aa:
            self.default_linetype = cv2.LINE_AA
        else:
            self.default_linetype = cv2.LINE_4
