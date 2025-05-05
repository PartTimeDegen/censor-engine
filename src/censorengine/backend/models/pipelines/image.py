from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import itertools

import numpy as np

from censorengine.backend.models.pipelines.image_components.compile_masks import (
    ImageComponentCompileMasks,
)
from censorengine.backend.models.pipelines.image_components.generate_censors import (
    ImageComponentGenerateCensors,
)
from censorengine.backend.models.pipelines.image_components.generate_parts import (
    ImageComponentGenerateParts,
)
from censorengine.backend.models.tools.debugger import (
    Debugger,
    DebugLevels,
)

from censorengine.backend.constants.typing import CVImage
from censorengine.backend.models.structures.detected_part import Part
from censorengine.backend.models.config import Config

from censorengine.libs.detector_library.catalogue import (
    enabled_detectors,
    enabled_determiners,
)
from censorengine.lib_models.detectors import DetectedPartSchema
from uuid import uuid4, UUID
from censorengine.backend.models.tools.dev_tools import DevTools


@dataclass(slots=True)
class ImageProcessor(
    ImageComponentGenerateParts,
    ImageComponentCompileMasks,
    ImageComponentGenerateCensors,
):
    file_image: CVImage
    config: Config
    debug_level: DebugLevels = DebugLevels.NONE
    dev_tools: DevTools | None = field(default=None)

    # Internals
    force_png: bool = False

    detected_parts: list[DetectedPartSchema] = field(init=False, default_factory=list)
    extracted_information: dict[str, str] = field(init=False, default_factory=dict)

    image_parts: list[Part] = field(init=False, default_factory=list)

    empty_mask: CVImage = field(init=False)
    file_original_image: CVImage = field(init=False)
    file_uuid: UUID = field(init=False)

    debugger: Debugger = field(init=False)
    duration: float = field(init=False)

    def __post_init__(self):
        Part.part_id = itertools.count(start=1)
        self.detected_parts.clear()
        self.extracted_information.clear()
        self.file_uuid = uuid4()
        self.file_original_image = self.file_image.copy()

        # Debug
        self.debugger = Debugger("Image Processor", level=self.debug_level)
        self.debugger.time_total_start()
        self.debugger.display_onnx_info()

        # Empty Mask
        self.debugger.time_start("Create Empty Mask")
        self.empty_mask = self._create_empty_mask()
        self.debugger.time_stop()

        # Determine Image # TODO: Find Some Models I can Run
        # self.debugger.time_start("Determine Image")
        # self.extracted_information = {
        #     determiner.model_name: determiner.determine_image(self.file_images)
        #     for determiner in enabled_determiners
        # }
        # self.debugger.time_stop()

        # Detect Parts for Image
        self.debugger.time_start("Detect Parts")
        self._detect_parts()
        self.debugger.time_stop()

    # Post Init Helper Functions
    def _detect_parts(self):
        with ThreadPoolExecutor() as executor:
            detected_parts = list(
                executor.map(
                    lambda detector: detector.detect_image(self.file_image),
                    enabled_detectors,
                )
            )

        self.detected_parts = list(itertools.chain(*detected_parts))
        self.debugger.time_stop()

    # Dev Tools
    def _decompile_masks(
        self,
        subfolder: str | None = None,
        iter_part: Part | None = None,
    ):
        if self.dev_tools:
            self.dev_tools.dev_decompile_masks(
                self.image_parts if not iter_part else iter_part,
                subfolder=subfolder,
            )

    # Private
    def _create_empty_mask(self, inverse: bool = False):
        """
        This function acts as a factory for empty masks, due to 1) bad copying
        issues, and 2) because it's not a one-liner

        :param bool inverse: Inverses the mask to make it white on black, not black on white, defaults to False

        # TODO: Cache this if it's made

        """
        if inverse:
            return Part.normalise_mask(
                np.ones(self.file_image.shape, dtype=np.uint8) * 255  # type: ignore
            )
        return Part.normalise_mask(np.zeros(self.file_image.shape, dtype=np.uint8))

    # Public
    def return_output(self):
        return self.file_image

    def generate_parts_and_shapes(self):
        # Create Parts
        self._decompile_masks("00_stage_base_00_create_part")
        self.debugger.time_start("Create Parts")
        self.image_parts = self._create_parts(
            self.config,
            self._create_empty_mask(),
            self.file_uuid,
            self.detected_parts,
        )

        self.debugger.time_stop()
        self._decompile_masks("00_stage_result_00_create_part")

        # Merge Parts
        self._decompile_masks("00_stage_base_01_merged")
        self.debugger.time_start("Merge Parts")
        self.image_parts = self._merge_parts_if_in_merge_groups(self.image_parts)
        self.debugger.time_stop()
        self._decompile_masks("00_stage_result_01_merged")

        # Handle More Advanced Parts (i.e., Bars and Joints)
        self._decompile_masks("00_stage_base_02_advanced")
        self.debugger.time_start("Handle Advanced Shapes")
        self.image_parts = self._apply_and_generate_mask_shapes(
            self._create_empty_mask(),
            self.image_parts,
        )
        self.debugger.time_stop()
        self._decompile_masks("00_stage_result_02_advanced")

    def compile_masks(self):
        # Test Parts for Overlap
        self.debugger.time_start("Process State Logic")
        self.image_parts = self._process_state_logic_for_masks(self.image_parts)
        self.debugger.time_stop()

    def apply_censors(self):
        # Generate and Apply Reverse Censor
        self.debugger.time_start("Generate Reverse Censor")
        self.file_image = self._handle_reverse_censor(
            self.config.reverse_censor.censors,
            self._create_empty_mask(inverse=True),
            self.image_parts,
            self.file_image,
        )
        self.debugger.time_stop()

        # Apply Censors
        self.debugger.time_start("Apply Censors")
        self.file_image = self._apply_censors(
            self.image_parts,
            self.file_image,
            self.force_png,
        )
        self.debugger.time_stop()

        # DEBUG: Times
        self.debugger.time_total_end()
        self.debugger.display_times()

    def start(self):
        self.generate_parts_and_shapes()
        self.compile_masks()
        self.apply_censors()

    # Util
    def get_part_list(self) -> dict[str, Part]:
        counter = 0
        final_dict = {}
        last_part = ""
        sorted_parts_list = sorted(
            self.image_parts, key=lambda part: (part.part_name, part.part_id)
        )
        for part in sorted_parts_list:
            if last_part == part.part_name:
                final_dict[f"{part.part_name}_{counter}"] = part
                counter += 1
            else:
                counter = 0
                final_dict[f"{part.part_name}_{counter}"] = part

        return final_dict

    def get_duration(self) -> float:
        return sum([time.duration for time in self.debugger.time_logger])
