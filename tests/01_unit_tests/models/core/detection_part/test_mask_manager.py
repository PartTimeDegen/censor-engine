import numpy as np
import pytest

from censor_engine.models.core.detection_part._mask_manager import MaskManager


class DummyMask:
    def generate(self, context):
        return np.full(context.empty_mask.shape, 100, dtype=np.uint8)


class DummyProtectionMask:
    def generate(self, context):
        return np.full(context.empty_mask.shape, 50, dtype=np.uint8)


@pytest.fixture
def registry(monkeypatch):
    masks = {
        "base": DummyMask,
        "protected": DummyProtectionMask,
    }

    monkeypatch.setattr(
        "censor_engine.models.core.detection_part._mask_manager.MaskRegistry.get_all",
        lambda: masks,
    )

    monkeypatch.setattr(
        "censor_engine.models.core.detection_part._mask_manager.Mask.create_empty_mask",
        lambda shape: np.zeros(shape, dtype=np.uint8),
    )

    return masks


class TestMaskManager:
    def test_initializes_masks(self, registry):
        manager = MaskManager(
            mask_name="base",
            protection_mask_name="protected",
            image_shape=(4, 4),
        )

        expected = np.full((4, 4), 100, dtype=np.uint8)

        assert np.array_equal(manager._original_mask, expected)
        assert np.array_equal(manager._current_mask, expected)
        assert manager._layers_of_mask == [manager._original_mask]
        assert isinstance(manager._obj_mask, DummyMask)
        assert isinstance(manager._obj_mask_protected, DummyProtectionMask)

    def test_uses_copy_of_base_mask_when_no_protection_mask(self, registry):
        manager = MaskManager(
            mask_name="base",
            protection_mask_name=None,
            image_shape=(2, 2),
        )

        assert isinstance(manager._obj_mask_protected, DummyMask)
        assert manager._obj_mask is not manager._obj_mask_protected

    class TestMethods:
        class TestGetMaskClass:
            def test_returns_instance(self, registry):
                mask = MaskManager.get_mask_class("base")

                assert isinstance(mask, DummyMask)

            def test_raises_for_unknown_mask(self, monkeypatch):
                monkeypatch.setattr(
                    "censor_engine.models.core.detection_part._mask_manager.MaskRegistry.get_all",
                    dict,
                )

                with pytest.raises(
                    ValueError, match="Mask missing does not Exist"
                ):
                    MaskManager.get_mask_class("missing")

        class TestMaskOperations:
            def test_add_to_current_mask(self, registry):
                manager = MaskManager(
                    mask_name="base",
                    protection_mask_name=None,
                    image_shape=(2, 2),
                )

                manager._current_mask = np.full((2, 2), 100, dtype=np.uint8)
                manager.add_to_current_mask(
                    np.full((2, 2), 50, dtype=np.uint8)
                )

                expected = np.full((2, 2), 150, dtype=np.uint8)
                assert np.array_equal(manager._current_mask, expected)

            def test_add_to_current_mask_saturates(self, registry):
                manager = MaskManager(
                    mask_name="base",
                    protection_mask_name=None,
                    image_shape=(2, 2),
                )

                manager._current_mask = np.full((2, 2), 250, dtype=np.uint8)
                manager.add_to_current_mask(
                    np.full((2, 2), 20, dtype=np.uint8)
                )

                assert np.all(manager._current_mask == 255)

            def test_subtract_from_current_mask(self, registry):
                manager = MaskManager(
                    mask_name="base",
                    protection_mask_name=None,
                    image_shape=(2, 2),
                )

                manager._current_mask = np.full((2, 2), 100, dtype=np.uint8)
                manager.subtract_from_current_mask(
                    np.full((2, 2), 40, dtype=np.uint8)
                )

                expected = np.full((2, 2), 60, dtype=np.uint8)
                assert np.array_equal(manager._current_mask, expected)

            def test_subtract_from_current_mask_saturates(self, registry):
                manager = MaskManager(
                    mask_name="base",
                    protection_mask_name=None,
                    image_shape=(2, 2),
                )

                manager._current_mask = np.full((2, 2), 10, dtype=np.uint8)
                manager.subtract_from_current_mask(
                    np.full((2, 2), 50, dtype=np.uint8)
                )

                assert np.all(manager._current_mask == 0)

            def test_compile_base_masks_adds_all_layers(self, registry):
                manager = MaskManager(
                    mask_name="base",
                    protection_mask_name=None,
                    image_shape=(2, 2),
                )

                manager._current_mask = np.zeros((2, 2), dtype=np.uint8)
                manager._layers_of_mask = [
                    np.full((2, 2), 10, dtype=np.uint8),
                    np.full((2, 2), 20, dtype=np.uint8),
                    np.full((2, 2), 30, dtype=np.uint8),
                ]

                manager.compile_base_masks()

                expected = np.full((2, 2), 60, dtype=np.uint8)
                assert np.array_equal(manager._current_mask, expected)
