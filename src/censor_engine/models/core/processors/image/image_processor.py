from dataclasses import dataclass, field
from uuid import uuid4

import cv2

from censor_engine.models.core.detection_part.detection_part import PartList
from censor_engine.models.core.processors.image.model_manager import (
    ModelManager,
)
from censor_engine.models.core.processors.image.part_generation._state_mechanism import (
    PartStateMechanism,
)
from censor_engine.models.libraries.ai_models.output_schemas import (
    DetectorOutput,
)
from censor_engine.typing import Image

from .censor_generation.censor_manager import (
    CensorManager,
)
from .detection.detection_manager import DetectionManager
from .part_generation.part_manager import (
    PartManager,
)
from .post_part_additionals.additionals_manager import AdditionalsManager


@dataclass(slots=True)
class ImageProcessor:
    _detection_manager: DetectionManager = field(
        default_factory=DetectionManager
    )
    _part_manager: PartManager = field(default_factory=PartManager)
    _additionals_manager: AdditionalsManager = field(
        default_factory=AdditionalsManager
    )
    _censor_manager: CensorManager = field(default_factory=CensorManager)

    def start_image_process(
        self,
        input_image: Image,
        model_manager: ModelManager,
    ) -> Image:
        # Detection
        detected_outputs: list[DetectorOutput] = (
            self._detection_manager.run_detection_pipeline(
                input_image=input_image,
                model_manager=model_manager,
            )
        )

        # Handle Masks
        # NOTE: For example if YOLO_PERSON_MASK is used over YOLO_PERSON,
        #       this means that the mask is preferred instead of the bbox.
        #       This is done with the suffix "_MASK".
        for detected_output in detected_outputs:
            # Short Cuts
            label = detected_output.label
            enabled_parts = model_manager.config.detection.enabled_parts
            if (
                enabled_parts is None
                or label is None
                or (label + "_MASK") not in enabled_parts
            ):
                continue

            detected_output.label = label + "_MASK"

        # Part Creation
        list_of_parts: PartList = (
            self._part_manager.run_part_generation_pipeline(
                detected_outputs,
                model_manager.config,
                uuid4(),
                input_image.shape[:2],
            )
        )

        # Post-Part Additionals
        # NOTE: This handles stuff that makes more sense after the parts have
        #       been created e.g., Depth makes more sense after the merge
        #       method has compiled the detections.
        processed_list_of_parts = (
            self._additionals_manager.run_additionals_pipeline(
                list_of_parts,
                model_manager.config,
            )
        )
        if (
            model_manager.config.engine_settings.extra_ai_model_features.depth
            is not None
        ):
            processed_list_of_parts.parts = (
                PartStateMechanism().handle_mask_overlaps_based_on_part_state(
                    processed_list_of_parts.parts
                )
            )

        # Censor Application
        return self._censor_manager.run_censor_generation_pipeline(
            input_image,
            processed_list_of_parts.parts,
        )
