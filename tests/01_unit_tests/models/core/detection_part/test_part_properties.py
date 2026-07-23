from censor_engine.models.core.detection_part._part_properties import (
    PartProperties,
)
from censor_engine.models.libraries.configs.config import Config
from censor_engine.models.libraries.detectors.schemas import (
    AbsoluteBBox,
    DetectorOutput,
)


class TestPartProperties:
    def test_initiate(
        self,
        detector_output: DetectorOutput,
        config_with_parts: Config,
    ):

        PartProperties(
            detector_output=detector_output, config=config_with_parts
        )

    class TestGeneratedFields:
        class TestCorrectedBox:
            def test_baseline(
                self,
                detection_properties: PartProperties,
                bbox: AbsoluteBBox,
            ):
                dp = detection_properties
                assert dp.bbox == bbox
                assert dp.bbox.centre == bbox.centre
                assert dp.bbox.area == bbox.area

            def test_margin(
                self,
                detection_properties_extras: PartProperties,
                bbox: AbsoluteBBox,
            ):
                dp = detection_properties_extras

                expected_change = round(bbox.height * 0.2) // 2
                lower = bbox.x1 - expected_change
                upper = bbox.x2 + expected_change
                expected_bbox = AbsoluteBBox(
                    x1=lower, y1=lower, x2=upper, y2=upper
                )

                assert dp.bbox == expected_bbox
                assert dp.bbox.centre == expected_bbox.centre
                assert dp.bbox.area == expected_bbox.area

        class TestIsMerged:
            def test_baseline(self, detection_properties: PartProperties):
                dp = detection_properties
                assert dp.is_merged == False

            def test_no_merge(
                self, detection_properties_extras: PartProperties
            ):
                dp = detection_properties_extras
                assert dp.is_merged == True

        class TestGroupMergeID:
            def test_baseline(self, detection_properties: PartProperties):
                dp = detection_properties
                assert dp.group_merge_id == None

            def test_no_merge_group(
                self, detection_properties_extras: PartProperties
            ):
                dp = detection_properties_extras
                assert dp.group_merge_id == 1

        class TestGroupMerge:
            def test_baseline(self, detection_properties: PartProperties):
                dp = detection_properties
                assert dp.group_merge == []

            def test_no_merge_group(
                self, detection_properties_extras: PartProperties, groups
            ):
                dp = detection_properties_extras
                assert dp.group_merge == groups[0]

        class TestGroupPersistID:
            def test_baseline(self, detection_properties: PartProperties):
                dp = detection_properties
                assert dp.group_persist_id == None

            def test_no_persist_group(
                self, detection_properties_extras: PartProperties
            ):
                dp = detection_properties_extras
                assert dp.group_persist_id == 1

        class TestGroupPersist:
            def test_baseline(self, detection_properties: PartProperties):
                dp = detection_properties
                assert dp.group_persist == []

            def test_no_persist_group(
                self, detection_properties_extras: PartProperties, groups
            ):
                dp = detection_properties_extras
                assert dp.group_persist == groups[0]
