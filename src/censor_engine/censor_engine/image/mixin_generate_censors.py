import cv2
import numpy as np

from censor_engine.typing import CVImage
from censor_engine.detected_part import Part
from censor_engine.models.structs import Censor
from censor_engine.libs.styles import style_catalogue


class MixinGenerateCensors:
    def _handle_reverse_censor(
        self,
        reverse_censors: list[Censor],
        inverse_empty_mask: CVImage,
        parts: list[Part],
        file_image: CVImage,
    ) -> CVImage:
        if not reverse_censors:
            return file_image

        # Create Mask
        base_mask_reverse = inverse_empty_mask
        for part in parts:
            base_mask_reverse = cv2.subtract(base_mask_reverse, part.mask)

        # Apply Censors
        contour = cv2.findContours(
            image=base_mask_reverse,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        for censor in reverse_censors[::-1]:
            censor_object = style_catalogue[censor.function]()
            censor_object.change_linetype(enable_aa=False)
            censor_object.using_reverse_censor = True

            arguments = [
                file_image,
                contour,
            ]

            file_image = censor_object.apply_style(
                *arguments,
                **censor.args,
            )
        return file_image

    def _apply_censors(self, parts: list[Part], file_image, force_png: bool) -> CVImage:
        parts = sorted(parts, key=lambda x: (x.part_settings.state, x.part_name))

        working_image = file_image.copy()
        for part in parts:
            if not part.part_settings.censors or not part:
                continue

            # if part.use_global_area:
            part_contour = cv2.findContours(
                image=part.mask,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )  # TODO: reduce image size to just part

            # Gather default args
            arguments = {
                "image": working_image,
                "contour": part_contour,
            }

            # Reversed to represent YAML order
            for censor in part.part_settings.censors[::-1]:
                # Acquire Function
                censor_object = style_catalogue[censor.function]()

                # Turn on AA
                censor_object.change_linetype(enable_aa=True)

                # Handle Args
                if censor_object.style_type == "dev":
                    arguments["part"] = part
                elif arguments.get("part"):
                    arguments.pop("part")

                # Apply Censor
                arguments["image"] = censor_object.apply_style(
                    **arguments,
                    **censor.args,
                )

                # Forces PNG if the Censor Requires it
                if censor_object.force_png:
                    self.force_png = censor_object.force_png

            # Apply Potential Feather Fade
            if not part.part_settings.fade_percent:
                working_image = arguments["image"]
                continue

            # # Get Mask
            contour_mask = cv2.drawContours(
                np.zeros(file_image.shape, dtype=np.uint8),
                part_contour[0],
                -1,
                (255, 255, 255),
                -1,
            )

            # # Convert Mask to Right Format
            contour_mask = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY)
            contour_mask = contour_mask.astype(float) / 255.0

            # # Calculate Feather Effect based on Contour
            _, _, w, h = cv2.boundingRect(part_contour[0][0])
            fade_percent = np.clip(int(part.part_settings.fade_percent), 0, 100)
            max_dim = max(w, h)
            feathering_amount = int((fade_percent / 100.0) * max_dim)
            kernel_size = min(51, max(3, feathering_amount // 2 * 2 + 1))

            # # Apply Gaussian blur for feathering
            contour_mask = cv2.erode(
                contour_mask,
                np.ones((kernel_size, kernel_size), np.uint8),
                iterations=1,
            ).astype(float)
            feathered_mask = cv2.GaussianBlur(
                contour_mask, (kernel_size, kernel_size), 0
            )

            # # Convert the mask back to 3-channel for blending
            feathered_mask = cv2.merge([feathered_mask] * 3)  # type: ignore

            # # Blend the images using the feathered mask
            feathered_mask *= 1.0
            merged_image = (
                feathered_mask * working_image + (1 - feathered_mask) * file_image
            )

            working_image = merged_image

        return working_image
