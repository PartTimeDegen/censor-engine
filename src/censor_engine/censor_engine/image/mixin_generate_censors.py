from dataclasses import fields
from typing import TYPE_CHECKING

import cv2
import numpy as np

from censor_engine.api.effects import EffectContext
from censor_engine.detected_part import Part
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.enums import EffectType
from censor_engine.models.structs import Censor, Mixin
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, TypeMask

if TYPE_CHECKING:
    from censor_engine.models.lib_models.effects.base import Effect

effects = EffectRegistry.get_all()


def get_contours_from_mask(mask: TypeMask) -> list[Contour]:
    """
    This is a helper function to normalise the contours from a mask and
    provide them as a Contour object.

    :param Mask mask: Mask
    :return list[Contour]: List of Contours
    """
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return [
        Contour(
            points=cnt,
            hierarchy=hierarchy[0][i] if hierarchy is not None else None,
        )
        for i, cnt in enumerate(contours)
    ]


def contours_to_mask(
    contours: list[Contour],
    image_mask: tuple[int, int],
    fill_value: int = 255,
) -> TypeMask:
    """
    This helper function converts the contours into a Mask.

    :param list[Contour] contours: List of Contours
    :param tuple[int, int] image_mask: Mask mask
    :param int fill_value: Value that's the background value, defaults to 255
    :return Mask: Mask
    """
    mask = np.zeros(image_mask, dtype=np.uint8)
    pts = [c.points for c in contours]
    cv2.drawContours(mask, pts, contourIdx=-1, color=fill_value, thickness=-1)  # type: ignore
    return mask


class MixinGenerateCensors(Mixin):
    """
    This Mixin handles the generation of the censors, both normal and reverse.

    """

    def _handle_reverse_censor(
        self,
        reverse_censors: list[Censor],
        inverse_empty_mask: TypeMask,
        parts: list[Part],
        file_image: Image,
    ) -> Image:
        """
        This method handles the generation of the reverse censor.

        How the method works, an inverse mask is created then the list of masks
        is subtracted from the inverse.

        :param list[Censor] reverse_censors: List of censors
        :param Mask inverse_empty_mask: Mask that's entirely white
        :param list[Part] parts: List of parts
        :param Image file_image: Original file image
        :return Image: Output image
        """
        # Skip if not Used
        if not reverse_censors:
            return file_image

        # Create Mask
        base_mask_reverse = inverse_empty_mask
        for part in parts:
            base_mask_reverse = cv2.subtract(base_mask_reverse, part.mask)

        # Apply Censors
        contours = get_contours_from_mask(base_mask_reverse)
        mask_norm = cv2.merge([base_mask_reverse] * 3)  # type: ignore
        for censor in reverse_censors[::-1]:
            # Get Censor Object
            censor_object = effects[censor.effect]()
            censor_object.change_linetype(enable_aa=False)
            censor_object.using_reverse_censor = True

            # Handle Fields from EffectContext that All Effects Have
            param_fields = {field.name for field in fields(EffectContext)}
            filtered_params = {
                key: value
                for key, value in censor.parameters.items()
                if key in param_fields
            }

            file_image = censor_object.internal_run_effect(
                EffectContext(
                    image=file_image,
                    contours=contours,
                    mask=mask_norm.copy(),
                    part=None,
                    part_list=[],
                    **filtered_params,
                ),
                **{
                    key: value
                    for key, value in censor.parameters.items()
                    if key not in filtered_params
                },
            )
        return file_image

    def _apply_censors(
        self,
        parts: list[Part],
        file_image: Image,
    ) -> tuple[Image, bool]:
        """
        This method handles the censoring of the files.

        The method sorts the part by state and name to ensure overlaps are
        properly handled, then the method will iterate through the censors.

        :param list[Part] parts: List of parts
        :param Image file_image: Input image
        :return tuple[Image, bool]: Output image and a flag for forcing the
            image to PNG (for transparency)
        """
        # Sort Parts Based on State and Name
        parts = sorted(
            parts,
            key=lambda x: (x.part_settings.state, x.part_name),
        )

        # Iterate through the Censors
        force_png = False
        working_image = file_image.copy()
        for part in parts:
            if not part.part_settings.censors:
                continue

            # Get Context Stuff from Part to Help Make Censors
            part_contours = get_contours_from_mask(part.mask)
            mask = contours_to_mask(part_contours, working_image.shape[:2])  # type: ignore
            mask_norm = cv2.merge([mask] * 3)  # type: ignore

            # Go Through the Censors of the Part
            for censor in part.part_settings.censors[::-1]:
                # Process the Censor Effects
                censor_object: Effect = effects[censor.effect]()
                censor_object.change_linetype(enable_aa=True)
                force_png = censor_object.force_png

                # Special Argument for Dev-type Effects where Full Awareness
                # is Required
                parts_list = (
                    parts
                    if censor_object.effect_type == EffectType.DEV
                    else None
                )

                # Handle Fields from EffectContext that All Effects Have
                param_fields = {field.name for field in fields(EffectContext)}
                filtered_params = {
                    key: value
                    for key, value in censor.parameters.items()
                    if key in param_fields
                }

                # Apply Effect
                working_image = censor_object.internal_run_effect(
                    EffectContext(
                        image=working_image.copy(),
                        contours=part_contours,
                        mask=mask_norm.copy(),
                        part=part,
                        part_list=parts_list,
                        **filtered_params,
                    ),
                    **{
                        key: value
                        for key, value in censor.parameters.items()
                        if key not in filtered_params
                    },
                )

            # === Feather (Gaussian) per-object glow ===
            if not part.part_settings.fade_percent:
                continue

            fade_factor = np.clip(
                part.part_settings.fade_percent / 100.0,
                0,
                1,
            )
            mask_bin = (mask > 0).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(mask_bin)
            output_mask = np.zeros_like(mask, dtype=np.float32)

            for label_id in range(1, num_labels):
                obj_mask = (labels == label_id).astype(np.uint8)
                if obj_mask.sum() == 0:
                    continue

                # Distance to background (edges)
                dist = cv2.distanceTransform(obj_mask, cv2.DIST_L2, 5)
                dist_norm = dist / dist.max() if dist.max() > 0 else dist

                # Optional: Gaussian effect
                spread = 3 + fade_factor * 5
                glow = np.exp(-((1 - dist_norm) ** 6) * spread)

                output_mask = np.maximum(output_mask, glow)

            # Convert to 0-255
            output_mask = np.clip(output_mask * 255, 0, 255).astype(np.uint8)

            # Prepare for blending
            mask_fade = cv2.merge([output_mask] * 3).astype(np.float32) / 255.0  # type: ignore
            blended = working_image.astype(
                np.float32,
            ) * mask_fade + file_image.astype(np.float32) * (1 - mask_fade)
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            working_image = blended

        return (working_image, force_png)
