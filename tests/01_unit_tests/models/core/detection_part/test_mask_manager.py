import numpy as np
import pytest

from censor_engine._typing import MaskImage
from censor_engine.models.core.detection_part._detection_properties import (
    DetectionProperties,
)
from censor_engine.models.libraries.configs.settings._detection import (
    _Margins,
)
from censor_engine.models.libraries.detectors.schemas import (
    AbsoluteBBox,
    DetectorOutput,
)

GROUPS = [
    ["boobs", "bum"],
    ["car", "vehicle"],
]


@pytest.fixture
def bbox() -> AbsoluteBBox:
    return AbsoluteBBox.from_xyxy((25, 25, 75, 75))


@pytest.fixture
def mask_empty() -> MaskImage:
    return np.zeros((100, 100), dtype=np.uint8)


@pytest.fixture
def mask_full() -> MaskImage:
    return np.ones((100, 100), dtype=np.uint8) * 255


@pytest.fixture
def masks(mask_empty, mask_full) -> list[MaskImage]:
    return [mask_empty, mask_full]


@pytest.fixture
def detector_output(
    bbox: AbsoluteBBox, masks: list[MaskImage]
) -> DetectorOutput:
    return DetectorOutput(
        bbox=bbox,
        masks=masks,  # assumes MaskImage is bytes-like in tests # type: ignore
        origin="test_origin",
        part_id=1,
        label=GROUPS[0][0],
        score=0.95,
    )


@pytest.fixture
def groups_merge() -> list[list[str]]:
    return GROUPS


@pytest.fixture
def groups_persistence() -> list[list[str]]:
    return GROUPS


@pytest.fixture
def detection_properties(
    detector_output: DetectorOutput,
) -> DetectionProperties:
    return DetectionProperties(
        detector_output=detector_output,
        margins=_Margins(),
        groups_merge=[],
        groups_persistance=[],
    )


@pytest.fixture
def detection_properties_extras(
    detector_output: DetectorOutput,
    groups_merge: list[list[str]],
    groups_persistence: list[list[str]],
) -> DetectionProperties:
    return DetectionProperties(
        detector_output=detector_output,
        margins=_Margins(height=0.2, width=0.2),
        groups_merge=groups_merge,
        groups_persistance=groups_persistence,
    )


class TestDetectionProperties:
    def test_initiate(
        self,
        detector_output: DetectorOutput,
        groups_merge: list[list[str]],
        groups_persistence: list[list[str]],
    ):
        DetectionProperties(
            detector_output=detector_output,
            margins=_Margins(height=0.2, width=0.2),
            groups_merge=groups_merge,
            groups_persistance=groups_persistence,
        )

    class TestGeneratedFields:
        class TestCorrectedBox:
            def test_baseline(
                self,
                detection_properties: DetectionProperties,
                bbox: AbsoluteBBox,
            ):
                dp = detection_properties
                assert dp._corrected_bbox == bbox
                assert dp._corrected_bbox.center == bbox.center
                assert dp._corrected_bbox.area == bbox.area

            def test_margin(
                self,
                detection_properties_extras: DetectionProperties,
                bbox: AbsoluteBBox,
            ):
                dp = detection_properties_extras

                expected_change = round(bbox.height * 0.2) // 2
                lower = bbox.x1 - expected_change
                upper = bbox.x2 + expected_change
                expected_bbox = AbsoluteBBox(
                    x1=lower, y1=lower, x2=upper, y2=upper
                )

                assert dp._corrected_bbox == expected_bbox
                assert dp._corrected_bbox.center == expected_bbox.center
                assert dp._corrected_bbox.area == expected_bbox.area

        class TestIsMerged:
            def test_baseline(self, detection_properties: DetectionProperties):
                dp = detection_properties
                assert dp._is_merged == False

            def test_no_merge(
                self, detection_properties_extras: DetectionProperties
            ):
                dp = detection_properties_extras
                assert dp._is_merged == True

        class TestGroupMergeID:
            def test_baseline(self, detection_properties: DetectionProperties):
                dp = detection_properties
                assert dp._group_merge_id == None

            def test_no_merge_group(
                self, detection_properties_extras: DetectionProperties
            ):
                dp = detection_properties_extras
                assert dp._group_merge_id == 1

        class TestGroupMerge:
            def test_baseline(self, detection_properties: DetectionProperties):
                dp = detection_properties
                assert dp._group_merge == []

            def test_no_merge_group(
                self, detection_properties_extras: DetectionProperties
            ):
                dp = detection_properties_extras
                assert dp._group_merge == GROUPS[0]

        class TestGroupPersistID:
            def test_baseline(self, detection_properties: DetectionProperties):
                dp = detection_properties
                assert dp._group_persist_id == None

            def test_no_persist_group(
                self, detection_properties_extras: DetectionProperties
            ):
                dp = detection_properties_extras
                assert dp._group_persist_id == 1

        class TestGroupPersist:
            def test_baseline(self, detection_properties: DetectionProperties):
                dp = detection_properties
                assert dp._group_persist == []

            def test_no_persist_group(
                self, detection_properties_extras: DetectionProperties
            ):
                dp = detection_properties_extras
                assert dp._group_persist == GROUPS[0]

    class TestMethods:
        class TestCorrectRelativeBoxSize:
            def test_baseline(self, detection_properties: DetectionProperties):
                dp = detection_properties

                fixed_box = dp._correct_relative_box_size(
                    dp._original_bbox,
                    dp.margins,
                )
                assert dp._corrected_bbox == fixed_box

            def test_with_margins(
                self, detection_properties_extras: DetectionProperties
            ):
                dp = detection_properties_extras

                fixed_box = dp._correct_relative_box_size(
                    dp._original_bbox,
                    dp.margins,
                )
                assert dp._corrected_bbox == fixed_box
