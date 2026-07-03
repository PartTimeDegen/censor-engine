from pathlib import Path

from censor_engine.models.core.path_manager._structs import ApprovedFormats
from censor_engine.models.core.path_manager.schemas import FileType


class TestApprovedFormats:
    def test_no_overlapping_types(self):
        # NOTE: This is just a confirmation test
        af = ApprovedFormats()
        assert af.image_formats & af.video_formats == frozenset()

    class TestProperties:
        class TestAllFormats:
            def test_baseline(self):
                af = ApprovedFormats()
                assert af.all_formats == af.image_formats | af.video_formats

    class TestMethods:
        class TestGetFormatType:
            def test_image(self):
                af = ApprovedFormats()
                assert af.get_format_type(Path("file.png")) == FileType.IMAGE

            def test_video(self):
                af = ApprovedFormats()
                assert af.get_format_type(Path("file.mp4")) == FileType.VIDEO

        class TestCheckIsApproved:
            def test_baseline(self):
                af = ApprovedFormats()
                assert af.check_is_approved(Path("file.png"))

            def test_folder(self):
                af = ApprovedFormats()
                assert af.check_is_approved(Path("folder")) == False

            def test_non_approved_file(self):
                af = ApprovedFormats()
                assert af.check_is_approved(Path("file.aaaaaa")) == False
