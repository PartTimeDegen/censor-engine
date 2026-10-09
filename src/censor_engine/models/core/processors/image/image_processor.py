from dataclasses import dataclass, field
from uuid import uuid4

from censor_engine._typing import Image
from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.core.path_manager.path_manager import PathManager
from censor_engine.models.core.processors.image.censor_generation.censor_manager import (  # noqa: E501
    CensorManager,
)
from censor_engine.models.core.processors.image.part_generation.part_manager import (  # noqa: E501
    PartManager,
)
from censor_engine.models.libraries.ai_models.output_schemas import (
    DetectorOutput,
)


@dataclass(slots=True)
class ImageProcessor:
    _part_manager: PartManager = field(default_factory=PartManager)
    _censor_manager: CensorManager = field(default_factory=CensorManager)

    def start_image_process(
        self,
        input_image: Image,
        path_manager: PathManager,
    ) -> Image:
        # Detection
        detected_outputs: list[DetectorOutput] = []

        # Part Creation
        list_of_parts: list[Part] = (
            self._part_manager.run_part_generation_pipeline(
                path_manager,
                detected_outputs,
                uuid4(),
                input_image.shape[:2],
            )
        )

        # Censor Application
        return self._censor_manager.run_censor_generation_pipeline(
            input_image,
            list_of_parts,
        )
