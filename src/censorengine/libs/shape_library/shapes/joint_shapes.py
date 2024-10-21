from dataclasses import dataclass
import cv2
import numpy as np

from censorengine.lib_models.shapes import JointShape


@dataclass
class JointBox(JointShape):
    shape_name: str = "joint_box"
    base_shape: str = "ellipse"
    single_shape: str = "box"

    def generate(self, dict_info, mask, box):
        mask_empty = np.zeros(dict_info["file_image"].shape, dtype=np.uint8)

        cont_rect = cv2.findContours(
            image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )
        shape_rect = np.int0(
            cv2.boxPoints(cv2.minAreaRect(np.vstack(cont_rect[0]).squeeze()))
        )

        mask = cv2.drawContours(
            mask_empty,
            [shape_rect],
            -1,
            (255, 255, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        return (
            mask,
            dict_info,
        )


@dataclass
class JointEllipse(JointShape):
    shape_name: str = "joint_ellipse"
    base_shape: str = "ellipse"
    single_shape: str = "ellipse"

    def generate(self, dict_info, mask, box):
        # Generate Base Contours
        mask_empty = np.zeros(dict_info["file_image"].shape, dtype=np.uint8)

        cont_base = cv2.findContours(
            image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )
        # Find Minimum Area Ellipse
        cont_flat = np.vstack(cont_base[0]).squeeze()
        shape_ellipse = cv2.fitEllipse(cont_flat)

        return (
            cv2.ellipse(mask_empty, shape_ellipse, (255, 255, 255), -1),
            dict_info,
        )


@dataclass
class RoundedJointBox(JointShape):
    shape_name: str = "rounded_joint_box"
    base_shape: str = "ellipse"
    single_shape: str = "rounded_box"

    def generate(self, dict_info, mask, box):
        mask_empty = np.zeros(dict_info["file_image"].shape, dtype=np.uint8)

        cont_rect = cv2.findContours(
            image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )
        shape_rect = np.int0(
            cv2.boxPoints(cv2.minAreaRect(np.vstack(cont_rect[0]).squeeze()))
        )

        mask_shapes = (
            cv2.drawContours(
                mask_empty,
                [shape_rect],
                -1,
                (255, 255, 255),
                -1,
                lineType=cv2.LINE_AA,
            ),
            dict_info,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        iterations = 2

        mask_changed = cv2.erode(
            mask_shapes[0],
            kernel,
            iterations=iterations >> 1,
        )
        mask_changed = cv2.dilate(
            mask_changed,
            kernel,
            iterations=iterations,
        )

        return (mask_changed, dict_info)


shapes = [
    JointBox,
    JointEllipse,
    RoundedJointBox,
]
