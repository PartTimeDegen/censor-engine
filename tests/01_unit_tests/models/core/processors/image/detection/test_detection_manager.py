from censor_engine.models.core.processors.image.detection.detection_manager import (
    DetectionManager,
)


class TestDetectionManager:
    class TestRunDetectionPipeline:
        def test_working(self):
            DetectionManager().run_detection_pipeline()
