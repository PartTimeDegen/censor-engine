from unittest.mock import PropertyMock, patch
from uuid import uuid4

import cv2
import numpy as np
import pytest

from censor_engine.models.libraries.masks._base_mask import Mask
from censor_engine.models.libraries.masks.bar_structs import (
    BarInfo,
    BoundingInfo,
)
from censor_engine.models.libraries.masks.schemas import MaskContext
from tests.fixtures.mask_contexts import (
    mask_context_inline_two_parts,
    mask_context_vertical,
)


@pytest.fixture
def context(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def bounding_info(mask_context_inline_two_parts):
    mask_obj = Mask()
    mask_obj.joint_mask = "JointEllipse"
    new_mask = mask_obj.generate_joint_mask(mask_context_inline_two_parts)
    cnt, _ = cv2.findContours(
        new_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    input_contours = max(cnt, key=cv2.contourArea)

    bi = BarInfo(False, False, False, False, uuid4())
    return BoundingInfo.create_bounding_info(
        contours=input_contours, bar_info=bi
    )


class TestBoundingInfo:
    class TestClassMethod:
        def test_ellipse_non_tight(self, mask_context_inline_two_parts):
            mask_obj = Mask()
            mask_obj.joint_mask = "JointEllipse"
            new_mask = mask_obj.generate_joint_mask(
                mask_context_inline_two_parts
            )
            cnt, _ = cv2.findContours(
                new_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            input_contours = max(cnt, key=cv2.contourArea)

            bi = BarInfo(False, False, False, False, uuid4())
            boi = BoundingInfo.create_bounding_info(
                contours=input_contours, bar_info=bi
            )
            assert boi.centre == (249, 250)
            assert boi.dimensions == (128, 390)
            assert boi.angle == -90
            assert boi.is_tight == False

        def test_ellipse_tight(self, mask_context_inline_two_parts):
            mask_obj = Mask()
            mask_obj.joint_mask = "JointEllipse"
            new_mask = mask_obj.generate_joint_mask(
                mask_context_inline_two_parts
            )
            cnt, _ = cv2.findContours(
                new_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            input_contours = max(cnt, key=cv2.contourArea)

            bi = BarInfo(False, False, True, False, uuid4())
            boi = BoundingInfo.create_bounding_info(
                contours=input_contours, bar_info=bi
            )
            assert boi.centre == (249, 250)
            assert boi.dimensions == (128, 390)
            assert boi.angle == 90
            assert boi.is_tight == True

        def test_rectangle_non_tight(self, mask_context_inline_two_parts):
            mask_obj = Mask()
            mask_obj.joint_mask = "JointBox"
            new_mask = mask_obj.generate_joint_mask(
                mask_context_inline_two_parts
            )
            cnt, _ = cv2.findContours(
                new_mask,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            input_contours = max(cnt, key=cv2.contourArea)

            bi = BarInfo(False, False, False, False, uuid4())
            boi = BoundingInfo.create_bounding_info(
                contours=input_contours, bar_info=bi
            )
            assert boi.centre == (248, 250)
            assert boi.dimensions == (40, 206)
            assert boi.angle == -90
            assert boi.is_tight == False

        def test_rectangle_tight(self, mask_context_inline_two_parts):
            mask_obj = Mask()
            mask_obj.joint_mask = "JointBox"
            new_mask = mask_obj.generate_joint_mask(
                mask_context_inline_two_parts
            )
            cnt, _ = cv2.findContours(
                new_mask,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            input_contours = max(cnt, key=cv2.contourArea)

            bi = BarInfo(False, False, True, False, uuid4())
            boi = BoundingInfo.create_bounding_info(
                contours=input_contours, bar_info=bi
            )
            assert boi.centre == (247, 374)
            # assert boi.dimensions == (40, 206) # (207, 1980)?
            assert boi.angle == 0.742
            # assert boi.is_tight ==  # Wrong?

        def test_did_not_use(self):
            with pytest.raises(ValueError):
                BoundingInfo(
                    centre=(0, 0),
                    dimensions=(0, 0),
                    angle=0,
                    contours=np.zeros((10, 10), dtype=np.int32),
                    bar_info=BarInfo(False, False, True, False, uuid4()),
                    is_tight=False,
                )

    class TestProperties:
        @pytest.mark.parametrize(
            "context, expected",
            [
                ("mask_context_inline_two_parts", True),
                ("mask_context_vertical", False),
            ],
            indirect=["context"],
        )
        def test_is_wider_than_taller_rectangle(
            self, context: MaskContext, expected: bool
        ):
            mask_obj = Mask()
            mask_obj.joint_mask = "JointEllipse"
            new_mask = mask_obj.generate_joint_mask(context)
            cnt, _ = cv2.findContours(
                new_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            input_contours = max(cnt, key=cv2.contourArea)

            bi = BarInfo(False, False, False, False, uuid4())
            boi = BoundingInfo.create_bounding_info(
                contours=input_contours,
                bar_info=bi,
            )

            assert boi.is_wider_than_taller_rectangle == expected

        @pytest.mark.parametrize(
            "long_direction, result", [(False, 128), (True, 390)]
        )
        def test_bar_thickness(
            self,
            long_direction: bool,
            result: int,
            bounding_info: BoundingInfo,
        ):
            boi = bounding_info
            boi.bar_info.is_long_direction = long_direction

            assert boi.bar_thickness == result

        @pytest.mark.parametrize(
            "force_hor, force_vert, result",
            [
                (False, False, False),
                (True, False, True),
                (False, True, True),
                (True, True, True),
            ],
        )
        def test_is_forced_direction(
            self,
            force_hor: bool,
            force_vert: bool,
            result: bool,
            bounding_info: BoundingInfo,
        ):
            boi = bounding_info
            boi.bar_info.force_horizontal = force_hor
            boi.bar_info.force_vertical = force_vert

            assert boi.is_forced_direction == result

    class TestPrivateMethods:
        @pytest.mark.parametrize(
            "tight_bar, is_taller_than_wider, angle, result_angle, result_dimensions",
            [
                # When Angle is less than snap
                (True, False, 0.1, 0.1, (128, 390)),
                (True, True, 0.1, 0.1, (128, 390)),
                (False, True, 0.1, 0.1, (128, 390)),
                (False, False, 0.1, 0.1, (390, 128)),  # Flipped Dim
                # When Angle is more than snap
                (True, False, 10.0, 10.0, (128, 390)),
                (True, True, 10.0, 10.0, (128, 390)),
                (False, True, 10.0, 100.0, (128, 390)),
                (False, False, 10.0, 10.0, (390, 128)),  # Flipped Dim
            ],
        )
        def test_fix_rectangle_axes(
            self,
            tight_bar: bool,
            is_taller_than_wider: bool,
            angle: float,
            bounding_info: BoundingInfo,
            result_angle: float,
            result_dimensions: tuple[float],
        ):
            boi = bounding_info

            boi.bar_info.is_tight_bar = tight_bar
            boi.angle = angle

            with patch.object(
                type(boi),
                "is_wider_than_taller_rectangle",
                new_callable=PropertyMock,
                return_value=is_taller_than_wider,
            ):
                boi._fix_rectangle_axes()

            assert boi.angle == result_angle
            assert boi.dimensions == result_dimensions

        @pytest.mark.parametrize(
            "is_long_direction, is_tight",
            [
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ],
        )
        def test_fix_long_direction_axes(
            self,
            is_long_direction: bool,
            is_tight: bool,
            bounding_info: BoundingInfo,
        ):
            boi = bounding_info
            boi.bar_info.is_long_direction = is_long_direction
            boi.is_tight = is_tight
            addon = is_long_direction or is_tight
            old_angle = boi.angle
            boi._fix_long_direction_axes()
            assert boi.angle == old_angle + 90 * addon

        @pytest.mark.parametrize(
            "angle, result",
            [
                (0.0, 0.0),
                (90.0, 90.0),
                (180.0, 0.0),
                (270.0, 90.0),
                (-90.0, 90.0),
                (-180.0, 0.0),
            ],
        )
        def test_normalise_angle(
            self, angle: float, result: float, bounding_info: BoundingInfo
        ):
            boi = bounding_info
            boi.angle = angle
            boi._normalise_angle()

            assert boi.angle == result

    class TestPublicMethods:
        def test_fix_angles(self): ...  # Done through others

        @pytest.mark.parametrize(
            "angle, result",
            [
                # Horizontal Snap
                (0.0, 0.0),
                (0.1, 0.0),
                (1.0, 0.0),
                (10.0, 10.0),
                # Vertical Snap
                (90.0, 90.0),
                (89.9, 90.0),
                (89.0, 90.0),
                (85.0, 85.0),
            ],
        )
        def test_snap_angle(
            self, angle: float, result: float, bounding_info: BoundingInfo
        ):
            boi = bounding_info
            boi.angle = angle
            boi.snap_angle()

            assert boi.angle == result

        @pytest.mark.parametrize(
            "force_hor, force_vert, result",
            [
                (False, False, None),
                (False, True, 391),
                (True, False, 129),
                (True, True, 129),
            ],
        )
        def test_handle_forced_directions(
            self,
            force_hor: bool,
            force_vert: bool,
            result: int | None,
            bounding_info: BoundingInfo,
        ):
            boi = bounding_info
            boi.bar_info.force_horizontal = force_hor
            boi.bar_info.force_vertical = force_vert

            assert boi.handle_forced_directions() == result
