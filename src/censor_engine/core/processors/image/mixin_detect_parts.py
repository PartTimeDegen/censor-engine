import itertools
from concurrent.futures import ThreadPoolExecutor

from censor_engine.models.caching import Cache
from censor_engine.models.caching.schemas import (
    AIOutputData,
    CacheData,
)
from censor_engine.models.config import Config
from censor_engine.models.lib_models.detectors.ai_models import AIModel
from censor_engine.models.lib_models.detectors.schemas import (
    DetectedPart,
)
from censor_engine.models.structs import Mixin
from censor_engine.typing import Image


class MixinDetectParts(Mixin):
    def __check_cache(
        self,
        frame: int | None,
        cache: Cache,
        detectors: list[AIModel],
    ) -> dict[AIModel, bool]:  # detector : is_in_cache

        return {
            detector: cache.check_for_frame(frame, detector.model_name)
            for detector in detectors
        }

    def __run_detector(
        self,
        detectors: list[AIModel],
        image: Image,
        config: Config,
    ) -> list[DetectedPart]:
        ai_settings = config.ai_settings
        detector_by_name = {
            detector.model_name: detector for detector in detectors
        }

        use_roi_focus = ai_settings.extras.focused_roi
        use_body_seg = ai_settings.extras.body_segmentation
        use_skin_part = "SKIN" in ai_settings.detections_enabled

        # Run YOLO First if Using Skin or ROI
        roi_output = None
        model_outputs: DetectedPart = []
        if use_roi_focus or use_body_seg or use_skin_part:
            roi_output = detector_by_name["YoloSeg"].detect_image(image)
            # TODO: Add PERSON label

        # Run NudeNet Next
        if roi_output:
            nudenet_output = [
                detector_by_name["NudeNet"].detect_image_with_roi(roi_output)  # type: ignore
            ]
        # TODO Rest need to be done, and also remove "Interface, it's not needed"

        # Then Run Body Segment

        # Run Rest of the Detectors

        with ThreadPoolExecutor() as executor:
            detected_parts = list(
                executor.map(
                    lambda detector: detector.detect_image(image=image),
                    detectors,
                ),
            )
        return list(itertools.chain(*detected_parts))

    def __get_already_cached_parts(
        self,
        frame: int | None,
        cache: Cache,
        cached_detectors: list[AIModel],
    ) -> list[DetectedPart]:
        return [
            part
            for detector in cached_detectors
            for part in cache.get_frame(frame, detector.model_name).output_data
        ]

    def __sort_outputs(
        self, all_parts: list[DetectedPart]
    ) -> list[DetectedPart]:
        bbox_parts = [part for part in all_parts if part.bbox is not None]
        not_bbox_parts = [part for part in all_parts if part.bbox is None]

        bbox_parts.sort(
            key=lambda part: (part.bbox[1], part.bbox[0]),  # type: ignore
        )

        sorted_parts = bbox_parts + not_bbox_parts

        for index, part in enumerate(sorted_parts, start=1):
            part.set_part_id(index)

        return sorted_parts

    def __cache_results(
        self,
        detectors: list[AIModel],
        all_parts: list[DetectedPart],
        frame: int | None,
        cache: Cache,
    ) -> None:
        model_names = {detector.model_name for detector in detectors}

        model_cache = CacheData(
            cache_data=[
                AIOutputData(
                    model_name=model_name,
                    output_data=[
                        part for part in all_parts if part.origin == model_name
                    ],
                )
                for model_name in model_names
            ],
        )

        cache.save_frame(frame, model_cache)

    def _detect_parts(
        self,
        frame: int | None,
        cache: Cache | None,
        detectors: list[AIModel],
        image: Image,
    ):
        if cache is None:
            msg = "Cache is missing!"
            raise TypeError(msg)

        # Get Cached and Non-cached Detectors
        verdicts = self.__check_cache(frame, cache, detectors)

        # Handle Missing Parts
        missing_detections: list[AIModel] = [
            k for k, v in verdicts.items() if not v
        ]
        print(missing_detections)
        detected_parts = self.__run_detector(missing_detections, image)

        # Handle Cached Parts
        cached_detectors: list[AIModel] = [
            detector for detector, verdict in verdicts.items() if verdict
        ]
        cached_parts = self.__get_already_cached_parts(
            frame, cache, cached_detectors
        )

        # Get and Sort Parts
        all_parts = self.__sort_outputs(detected_parts + cached_parts)

        # Cache Parts
        self.__cache_results(detectors, all_parts, frame, cache)

        return all_parts
