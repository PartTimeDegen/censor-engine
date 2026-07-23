import pytest

from censor_engine._typing import MaskImage
from censor_engine.models.core.detection_part._part_properties import (
    PartProperties,
)
from censor_engine.models.libraries.masks.schemas import MaskContext


@pytest.fixture
def mask_context_no_part(
    detection_properties_extras: PartProperties,
    mask_empty: MaskImage,
    single_part,
) -> MaskContext:
    return MaskContext(
        single_part,
        detection_properties_extras,
        mask_empty,
        mask_empty,
    )


@pytest.fixture
def mask_context_single_part(
    detection_properties_extras: PartProperties,
    mask_single_part,
    mask_empty: MaskImage,
    single_part,
) -> MaskContext:
    fixed_props = detection_properties_extras
    fixed_props.is_merged = False
    return MaskContext(
        single_part,
        fixed_props,
        mask_single_part,
        mask_empty,
    )


@pytest.fixture
def mask_context_inline_two_parts(
    detection_properties_extras: PartProperties,
    mask_two_parts_inline,
    mask_empty: MaskImage,
    single_part,
) -> MaskContext:
    return MaskContext(
        single_part,
        detection_properties_extras,
        mask_two_parts_inline,
        mask_empty,
    )


@pytest.fixture
def mask_context_diagonal(
    detection_properties_extras: PartProperties,
    mask_two_parts_diagonal,
    mask_empty: MaskImage,
    single_part,
) -> MaskContext:
    return MaskContext(
        single_part,
        detection_properties_extras,
        mask_two_parts_diagonal,
        mask_empty,
    )


@pytest.fixture
def mask_context_vertical(
    detection_properties_extras: PartProperties,
    mask_two_parts_vertical,
    mask_empty: MaskImage,
    single_part,
) -> MaskContext:
    return MaskContext(
        single_part,
        detection_properties_extras,
        mask_two_parts_vertical,
        mask_empty,
    )


@pytest.fixture
def mask_context_quad_square(
    detection_properties_extras: PartProperties,
    mask_two_parts_quad_square,
    mask_empty: MaskImage,
    single_part,
) -> MaskContext:
    return MaskContext(
        single_part,
        detection_properties_extras,
        mask_two_parts_quad_square,
        mask_empty,
    )


@pytest.fixture
def mask_context_three_triangle_top(
    detection_properties_extras: PartProperties,
    mask_three_triangle_top,
    mask_empty: MaskImage,
    single_part,
) -> MaskContext:
    return MaskContext(
        single_part,
        detection_properties_extras,
        mask_three_triangle_top,
        mask_empty,
    )


@pytest.fixture
def mask_context_three_triangle_bottom(
    detection_properties_extras: PartProperties,
    mask_three_triangle_bottom,
    mask_empty: MaskImage,
    single_part,
) -> MaskContext:
    return MaskContext(
        single_part,
        detection_properties_extras,
        mask_three_triangle_bottom,
        mask_empty,
    )
