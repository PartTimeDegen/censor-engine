import cv2
import numpy as np

from censor_engine.detected_part import Part
from censor_engine.libs.registries import StyleRegistry
from censor_engine.models.enums import StyleType
from censor_engine.models.lib_models.styles.base import Style
from censor_engine.models.structs import Censor, Mixin
from censor_engine.models.structs.contours import Contour
from censor_engine.typing import Image, Mask

styles = StyleRegistry.get_all()


def get_contours_from_mask(mask: Mask) -> list[Contour]:
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
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
    image_shape: tuple[int, int],
    fill_value: int = 255,
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = [c.points for c in contours]
    cv2.drawContours(mask, pts, contourIdx=-1, color=fill_value, thickness=-1)
    return mask


class MixinGenerateCensors(Mixin):
    def _handle_reverse_censor(
        self,
        reverse_censors: list[Censor],
        inverse_empty_mask: Mask,
        parts: list[Part],
        file_image: Image,
    ) -> Image:
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
            censor_object = styles[censor.style]()
            censor_object.change_linetype(enable_aa=False)
            censor_object.using_reverse_censor = True

            file_image = censor_object.internal_run_style(
                image=file_image,
                contours=contours,
                mask=mask_norm.copy(),
                part=None,
                **censor.parameters,
            )
        return file_image

    def _apply_censors(
        self,
        parts: list[Part],
        file_image,
    ) -> tuple[Image, bool]:
        parts = sorted(
            parts,
            key=lambda x: (x.part_settings.state, x.part_name),
        )
        force_png = False
        working_image = file_image.copy()
        for part in parts:
            if not part.part_settings.censors:
                continue

            part_contours = get_contours_from_mask(part.mask)
            mask = contours_to_mask(part_contours, working_image.shape[:2])  # type: ignore
            mask_norm = cv2.merge([mask] * 3)  # type: ignore

            for censor in part.part_settings.censors[::-1]:
                censor_object: Style = styles[censor.style]()
                censor_object.change_linetype(enable_aa=True)
                force_png = censor_object.force_png

                additional_args = {
                    **censor.parameters,
                    **(
                        {"part_list": parts}
                        if censor_object.style_type == StyleType.DEV
                        else {}
                    ),
                }

                working_image = censor_object.internal_run_style(
                    image=working_image.copy(),
                    contours=part_contours,
                    mask=mask_norm.copy(),
                    part=part,
                    **additional_args,
                )

            # === Feather Stuff === #

            # Apply Potential Feather Fade
            if not part.part_settings.fade_percent:
                continue

            # Use the first contour for mask and bounding box (adjust as needed)
            mask_norm = (mask_norm / 255).astype(np.float32)
            x, y, w, h = part_contours[0].as_bounding_box()
            fade_percent = np.clip(
                int(part.part_settings.fade_percent), 0, 100
            )
            max_dim = max(w, h)
            feathering_amount = int((fade_percent / 100.0) * max_dim)
            kernel_size = min(51, max(3, feathering_amount // 2 * 2 + 1))

            mask_norm = cv2.erode(
                mask_norm,  # type: ignore
                np.ones((kernel_size, kernel_size), np.uint8),
                iterations=1,
            ).astype(float)
            feathered_mask = cv2.GaussianBlur(
                mask_norm, (kernel_size, kernel_size), 0
            )

            feathered_mask = cv2.merge([feathered_mask] * 3)  # type: ignore

            feathered_mask = feathered_mask.astype(np.float32)
            working_image = working_image.astype(np.float32)
            file_image = file_image.astype(np.float32)

            merged_image = (
                feathered_mask * working_image
                + (1 - feathered_mask) * file_image
            )
            working_image = merged_image.astype(np.uint8)

        return (working_image, force_png)
