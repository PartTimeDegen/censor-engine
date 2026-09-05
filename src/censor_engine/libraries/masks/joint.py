import cv2
import numpy as np

from censor_engine._typing import MaskImage
from censor_engine.libraries.registries import MaskRegistry
from censor_engine.models.libraries.masks.masks import JointMask
from censor_engine.models.libraries.masks.schemas import MaskContext


@MaskRegistry.register()
class JointBox(JointMask):
    single_mask = "Box"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        # Get Contours
        cont_rect = self.get_contours(mask_context)
        if len(cont_rect) == 1:
            return self.generate_single_mask(mask_context)

        # Acquired from:
        # https://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
        rect = cv2.minAreaRect(np.vstack(cont_rect).squeeze())
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        return cv2.drawContours(
            image=mask_context.empty_mask,
            contours=[box],
            contourIdx=-1,
            **mask_context.cv2_standard_settings,
        )  # type: ignore


@MaskRegistry.register()
class JointEllipse(JointMask):
    single_mask = "Ellipse"

    def generate_mask(  # type: ignore
        self,
        mask_context: MaskContext,
        *,
        scale: float = 1.0,
    ) -> MaskImage:
        # Get Contours
        cont_rect = self.get_contours(mask_context)
        if len(cont_rect) == 1:
            return self.generate_single_mask(mask_context)

        # Find Minimum Area Ellipse
        cont_flat = np.vstack(cont_rect).squeeze()
        centre, axes, angle = cv2.fitEllipse(cont_flat)

        min_size = min(axes)
        new_axes = [int(scale * x) if x == min_size else int(x) for x in axes]

        # Fixed Types
        # NOTE: Amazingly, OpenCV doesn't give the same types for fitting an
        #       ellipse as when making one. Remarkable package.
        new_centre = [int(x) for x in centre]

        return cv2.ellipse(
            mask_context.empty_mask,  # type: ignore
            center=new_centre,
            axes=new_axes,
            angle=angle,
            startAngle=0,
            endAngle=360,
            **mask_context.cv2_standard_settings,
        )  # type: ignore


@MaskRegistry.register()
class Block(JointMask):
    single_mask = "Box"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        # Get Contours
        cont_rect = self.get_contours(mask_context)
        if len(cont_rect) == 1:
            return self.generate_single_mask(mask_context)

        x, y, w, h = cv2.boundingRect(np.vstack(cont_rect))  # type: ignore

        # Define the box points (4 corners)
        box = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype=np.int32,
        )

        return cv2.drawContours(
            image=mask_context.empty_mask,  # type: ignore
            contours=[box],
            contourIdx=-1,
            **mask_context.cv2_standard_settings,
        )  # type: ignore


@MaskRegistry.register()
class JointThickEllipse(JointMask):
    base_mask = "Ellipse"
    single_mask = "Ellipse"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        return JointEllipse().generate_mask(
            mask_context=mask_context, scale=1.5
        )


@MaskRegistry.register()
class JointThinEllipse(JointMask):
    base_mask = "Ellipse"
    single_mask = "Ellipse"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        return JointEllipse().generate_mask(
            mask_context=mask_context, scale=0.5
        )


@MaskRegistry.register()
class RoundedJointBox(JointMask):
    single_mask = "RoundedBox"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        mask = JointBox().generate_mask(mask_context)

        # Rounding Part
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        iterations = 2
        mask_changed = cv2.erode(
            mask,
            kernel,
            iterations=iterations >> 1,
        )
        return cv2.dilate(mask_changed, kernel, iterations=iterations)  # type: ignore
