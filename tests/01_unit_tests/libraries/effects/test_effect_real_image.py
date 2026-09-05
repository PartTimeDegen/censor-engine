from pathlib import Path

import cv2
import numpy as np
import pytest

from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.core.detection_part._part_properties import (
    PartProperties,
)
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from tests.helpers.test_data_handler import _get_test_data_path

file_path = Path(__file__)


def effect_context_test_image(
    detection_properties: PartProperties,
    # file_name: str,
):
    # image = cv2.imread(f"zz_test_images/{file_name}")
    image = cv2.imread("zz_test_images/1782038725813377.jpg")
    shape = image.shape[:2]
    min_dim = min(shape)

    # mask = np.ones(shape, dtype=np.uint8) * 255
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(
        mask,
        (shape[1] // 2, shape[0] // 2),
        int((min_dim // 2) * 0.9),
        255,
        thickness=-1,
    )

    return EffectContext(image, mask, detection_properties)


files = [path.parts[1] for path in list(Path("zz_test_images").glob("*"))]

effect_list = EffectRegistry.get_all().values()
effects_sorted = sorted(effect_list, key=lambda x: x.__name__)


# @pytest.mark.parametrize("file_name", files)
@pytest.mark.parametrize("effect", effects_sorted)
def test_effect(
    effect,
    # file_name: str,
    detection_properties: PartProperties,
):
    effect_obj = effect()
    folder = _get_test_data_path(file_path) / "custom"
    folder.mkdir(parents=True, exist_ok=True)
    # base_path = folder / f"{Path(effect.__name__)}_{file_name}"
    base_path = folder / f"{Path(effect.__name__)}"

    ec = effect_context_test_image(
        detection_properties,
        #    file_name,
    )

    effect_image = effect_obj.generate_effect(ec)
    cut_image = effect_obj.apply_effect_to_image(ec, effect_image)

    shape = cut_image.shape[:2]
    min_dim = min(shape)

    cv2.circle(
        cut_image,
        (shape[1] // 2, shape[0] // 2),
        int((min_dim // 2) * 0.9),
        (0, 0, 0),
        thickness=2,
    )

    cv2.imwrite(f"{base_path}.png", cut_image)
