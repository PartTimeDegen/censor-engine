from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from censor_engine.models.parts._schemas import PartNameType
from censor_engine.models.parts.part import Part


@pytest.fixture
def detector_output():
    output = MagicMock()
    output.label = "face"
    return output


@pytest.fixture
def config():
    return MagicMock()


@pytest.fixture
def image_shape():
    return (100, 200)


@pytest.fixture
def file_uuid():
    return uuid4()


@pytest.fixture
def properties():
    props = MagicMock()
    props.label = "face"
    props.part_id = "123"
    props.is_merged = False
    props.settings.mask = "mask"
    props.settings.protection_mask = "protection"
    return props


@pytest.fixture
def part(detector_output, config, file_uuid, image_shape, properties):
    with (
        patch(
            "censor_engine.models.parts.part.PartProperties",
            return_value=properties,
        ),
        patch(
            "censor_engine.models.parts.part.MaskManager",
        ),
    ):
        return Part(
            detector_output=detector_output,
            config=config,
            file_uuid=file_uuid,
            image_shape=image_shape,
        )


def test_part_initialization_creates_properties_and_masks(
    detector_output,
    config,
    file_uuid,
    image_shape,
    properties,
):
    with (
        patch(
            "censor_engine.models.parts.part.PartProperties",
            return_value=properties,
        ) as properties_mock,
        patch(
            "censor_engine.models.parts.part.MaskManager",
        ) as mask_mock,
    ):
        part = Part(
            detector_output=detector_output,
            config=config,
            file_uuid=file_uuid,
            image_shape=image_shape,
        )

    properties_mock.assert_called_once_with(
        detector_output=detector_output,
        config=config,
    )

    mask_mock.assert_called_once_with(
        mask_name=properties.settings.mask,
        protection_mask_name=properties.settings.protection_mask,
        image_shape=image_shape,
    )

    assert part.properties == properties
    assert part.masks == mask_mock.return_value


def test_part_raises_error_when_label_is_missing(
    config,
    file_uuid,
    image_shape,
):
    detector_output = MagicMock()
    detector_output.label = None

    with pytest.raises(TypeError, match="Missing Name"):
        Part(
            detector_output=detector_output,
            config=config,
            file_uuid=file_uuid,
            image_shape=image_shape,
        )


@pytest.mark.parametrize(
    "output, expected",
    [
        (PartNameType.NAME, "face"),
        (PartNameType.ID_AND_NAME, "123_face"),
        (PartNameType.ID_AND_NAME_AND_MERGED, "123_face_single"),
        (PartNameType.NAME_AND_MERGED, "123_face_single"),
    ],
)
def test_get_name(part, output, expected):
    assert part.get_name(output) == expected


def test_get_name_returns_name_for_unknown_output(part):
    assert part.get_name("invalid") == "face"


@pytest.mark.parametrize(
    "is_merged, expected",
    [
        (True, "123_face_merged"),
        (False, "123_face_single"),
    ],
)
def test_get_name_merged_state(part, is_merged, expected):
    part.properties.is_merged = is_merged

    assert part.get_name(PartNameType.ID_AND_NAME_AND_MERGED) == expected
