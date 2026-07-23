import cv2
import numpy as np
import pytest

from censor_engine.models.core.detection_part._mask_manager import (
    MaskContext,
    MaskManager,
)


class TestMaskManager:
    def test_init(self, mask_context: MaskContext):
        MaskManager("Bar", None, mask_context)

    class TestGeneratedFields:
        class TestCurrentMask:
            def test_baseline(self, mask_manager: MaskManager, mask_empty):

                assert isinstance(mask_manager.current_mask, np.ndarray)
                assert not np.array_equal(
                    mask_manager.current_mask, mask_empty
                )

        class TestLayersOfMask:
            def test_baseline(self):
                pytest.fail("Not implemented yet")

        class TestObjMask:
            def test_baseline(self):
                pytest.fail("Not implemented yet")

        class TestObjMaskProtected:
            def test_baseline(self):
                pytest.fail("Not implemented yet")

            def test_is_none(self):
                pytest.fail("Not implemented yet")

        class TestOriginalMask:
            def test_baseline(self):
                pytest.fail("Not implemented yet")

    class TestStaticMethods:
        class TestGetMaskClass:
            def test_baseline(self):
                pytest.fail("Not implemented yet")

            def test_does_not_exist(self):
                pytest.fail("Not implemented yet")

    class TestMethods:
        class TestCreateEmptyMask:
            def test_baseline(self):
                pytest.fail("Not implemented yet")

            def test_inverse(self):
                pytest.fail("Not implemented yet")

        class TestAddToCurrentMask:
            def test_baseline(self):
                pytest.fail("Not implemented yet")

        class TestSubtractFromCurrentMask:
            def test_baseline(self):
                pytest.fail("Not implemented yet")

        class TestCompileBaseMasks:
            def test_baseline(self):
                pytest.fail("Not implemented yet")

            def test_single(self):
                pytest.fail("Not implemented yet")
