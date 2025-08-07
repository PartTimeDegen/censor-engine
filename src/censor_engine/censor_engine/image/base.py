import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from censor_engine.censor_engine.tools.debugger import (
    Debugger,
    DebugLevels,
)
from censor_engine.censor_engine.tools.dev_tools import DevTools
from censor_engine.detected_part import Part
from censor_engine.libs.detectors import enabled_detectors
from censor_engine.models.config import Config
from censor_engine.models.lib_models.detectors import DetectedPartSchema
from censor_engine.typing import Image

from .mixin_compile_masks import MixinComponentCompile
from .mixin_generate_censors import MixinGenerateCensors
from .mixin_generate_parts import MixinGenerateParts


@dataclass(slots=True)
class ImageProcessor(
    MixinComponentCompile, MixinGenerateCensors, MixinGenerateParts
):
    file_image: Image
    config: Config
    debug_level: DebugLevels = DebugLevels.NONE
    dev_tools: DevTools | None = None
    _test_detection_output: list[DetectedPartSchema] | None = None

    # Internals
    _force_png: bool = False

    _detected_parts: list[DetectedPartSchema] = field(
        init=False, default_factory=list
    )
    _extracted_information: dict[str, str] = field(
        init=False, default_factory=dict
    )

    _image_parts: list[Part] = field(init=False, default_factory=list)

    _empty_mask: Image = field(init=False)
    _file_original_image: Image = field(init=False)
    _file_uuid: UUID = field(init=False)

    _debugger: Debugger = field(init=False)
    _duration: float = field(init=False)

    def __post_init__(self):
        self._detected_parts.clear()
        self._extracted_information.clear()
        self._file_uuid = uuid4()
        self._file_original_image = self.file_image.copy()

        # Debug
        self._debugger = Debugger("Image Processor", level=self.debug_level)
        self._debugger.time_total_start()
        self._debugger.display_onnx_info()

        # Empty Mask
        self._debugger.time_start("Create Empty Mask")
        self._debugger.time_stop()

        # Determine Image # TODO: Find Some Models I can Run
        # self.debugger.time_start("Determine Image")
        # self.extracted_information = {
        #     determiner.model_name: determiner.determine_image(self.file_images)
        #     for determiner in enabled_determiners
        # }
        # self.debugger.time_stop()

        # Detect Parts for Image
        self._debugger.time_start("Detect Parts")
        if self._test_detection_output:
            self._detected_parts = self._test_detection_output
            self._debugger.time_stop()
        else:
            self._detect_parts()
        self._debugger.time_stop()

    # Post Init Helper Functions
    def _detect_parts(self):
        with ThreadPoolExecutor() as executor:
            detected_parts = list(
                executor.map(
                    lambda detector: detector.detect_image(self.file_image),
                    enabled_detectors,
                )
            )

        all_parts = list(itertools.chain(*detected_parts))

        # Sort and Label ID Based on Position
        all_parts.sort(
            key=lambda part: (part.relative_box[1], part.relative_box[0])
        )

        for index, part in enumerate(all_parts, start=1):
            part.set_part_id(index)

        self._detected_parts = all_parts
        self._debugger.time_stop()

    # Dev Tools
    def _decompile_masks(
        self,
        subfolder: str | None = None,
        iter_part: Part | None = None,
    ):
        if self.dev_tools:
            self.dev_tools.dev_decompile_masks(
                self._image_parts if not iter_part else iter_part,
                subfolder=subfolder,
            )

    # Public
    def return_output(self):
        return self.file_image

    def generate_parts_and_shapes(self):
        # Create Parts
        self._decompile_masks("00_stage_base_00_create_part")
        self._debugger.time_start("Create Parts")
        self._image_parts = self._create_parts(
            self.config,
            self._file_uuid,
            self._detected_parts,
            self.file_image.shape,  # type: ignore
        )
        self._debugger.time_stop()
        self._decompile_masks("00_stage_result_00_create_part")

        # Merge Parts
        self._decompile_masks("00_stage_base_01_merged")
        self._debugger.time_start("Merge Parts")
        self._image_parts = self._merge_parts_if_in_merge_groups(
            self._image_parts
        )
        self._debugger.time_stop()
        self._decompile_masks("00_stage_result_01_merged")

        # Handle More Advanced Parts (i.e., Bars and Joints)
        self._decompile_masks("00_stage_base_02_advanced")
        self._debugger.time_start("Handle Advanced Shapes")
        self._image_parts = self._apply_and_generate_mask_shapes(
            self._image_parts
        )
        self._debugger.time_stop()
        self._decompile_masks("00_stage_result_02_advanced")

    def compile_masks(self):
        # Test Parts for Overlap
        self._debugger.time_start("Process State Logic")
        self._image_parts = self._process_state_logic_for_masks(
            self._image_parts
        )
        self._debugger.time_stop()

    def apply_censors(self):
        # Generate and Apply Reverse Censor
        self._debugger.time_start("Generate Reverse Censor")
        self.file_image = self._handle_reverse_censor(
            self.config.reverse_censor.censors,
            Part.create_empty_mask(
                self.file_image.shape,  # type: ignore
                inverse=True,
            ),
            self._image_parts,
            self.file_image,
        )
        self._debugger.time_stop()

        # Apply Censors
        self._debugger.time_start("Apply Censors")
        self.file_image, self._force_png = self._apply_censors(
            self._image_parts,
            self.file_image,
        )
        self._debugger.time_stop()

        # DEBUG: Times
        self._debugger.time_total_end()
        self._debugger.display_times()

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
            self._image_parts, key=lambda part: (part.part_name, part.part_id)
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
        return sum([time.duration for time in self._debugger.time_logger])
