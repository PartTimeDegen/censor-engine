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
    MixinComponentCompile,
    MixinGenerateCensors,
    MixinGenerateParts,
):
    """
    This class is the processor for image censoring, and also how the video
    processing is processed as well.

    The processor has multiple parts for the full outline however it has been
    separated into methods for readability and modularity.

    Parameters
    ----------
    :param Image file_image: Base Image file (or a frame if it's from
        VideoProcessor)
    :param Config config: Config file that contains the settings
    :param debug_level debug_level: Debugging levels, used to quickly utilise
        different grades of debugging
    :param DevTools dev_tools: Debugging tools class
    :param list[DetectedPartSchema] | None _test_detection_output: Private
        method used by tests to inject mock data that would be from the AI
        model(s)

    :param MixinComponentCompile: Mixin that contains the methods to compile
        stuff
    :param MixinGenerateCensors: Mixin to generate the censors
    :param MixinGenerateParts: Mixin to create and merge the parts

    """

    file_image: Image
    config: Config
    debug_level: DebugLevels = DebugLevels.NONE
    dev_tools: DevTools | None = None
    _test_detection_output: list[DetectedPartSchema] | None = None

    # Internals
    force_png: bool = False

    _detected_parts: list[DetectedPartSchema] = field(
        init=False,
        default_factory=list,
    )
    _extracted_information: dict[str, str] = field(
        init=False,
        default_factory=dict,
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
        self._debugger.display_onnx_info()

        # Detect Parts for Image
        if self._test_detection_output:
            self._detected_parts = self._test_detection_output
            self._debugger.time_stop()
        else:
            self._detect_parts()

    # Post Init Helper Functions
    def _detect_parts(self) -> None:
        """
        This function detects the parts using the detector dataclass.

        It utilises multi-threading to speed up when multiple detectors are
        used. In theory it should only work marginally do to the bottleneck of
        using the GPU (or CPU), however it's still a minor improvement.

        """
        with ThreadPoolExecutor() as executor:
            detected_parts = list(
                executor.map(
                    lambda detector: detector.detect_image(self.file_image),
                    enabled_detectors,
                ),
            )

        all_parts = list(itertools.chain(*detected_parts))

        # Sort and Label ID Based on Position
        all_parts.sort(
            key=lambda part: (part.relative_box[1], part.relative_box[0]),
        )

        for index, part in enumerate(all_parts, start=1):
            part.set_part_id(index)

        self._detected_parts = all_parts

    # Dev Tools
    def _decompile_masks(
        self,
        subfolder: str | None = None,
        iter_part: Part | None = None,
    ) -> None:
        """
        This is a dev method used to handle the logging of masks while the
        masks aren't compiled, allowing for analyse of each part of the
        pipeline.

        Currently not used.

        TODO: Make into a Utils class so it's out the way.

        :param str | None subfolder: Used to subfolders, defaults to None
        :param Part | None iter_part: [Can't remember], defaults to None
        """
        if self.dev_tools:
            self.dev_tools.dev_decompile_masks(
                iter_part if iter_part else self._image_parts,
                subfolder=subfolder,
            )

    # Public
    # # Getters and Setters
    def get_image_parts(self) -> list[Part]:
        """
        This function returns the image parts, used to access the private
        field.

        :return list[Part]: List of parts.
        """
        return self._image_parts

    def set_image_parts(self, parts: list[Part]) -> None:
        """
        Setter for image parts.

        :param list[Part] parts: List of Parts.
        """
        self._image_parts = parts

    def return_output(self) -> Image:
        """
        Returns the output of the processor.

        :return Image: Current image.
        """
        return self.file_image

    def generate_parts_and_shapes(self) -> None:
        """
        This method handles the generation of the parts and the shapes.

        Stages:
            1)  Create parts.
            2)  Filter parts that don't meet the minimum score threshold.
            3)  Merge parts based on the merge method and merge groups.
            4)  Apple the shape effects to the mask, handling more advanced
                parts as well which require more than one pass.

        """
        # Create Parts
        self._image_parts = self._create_parts(
            self.config,
            self._file_uuid,
            self._detected_parts,
            self.file_image.shape,
        )

        # Filter Parts
        self._image_parts = [
            part
            for part in self._image_parts
            if part.score >= part.minimum_score
        ]

        # Merge Parts
        self._image_parts = self._merge_parts(
            self._image_parts,
        )

        # Handle More Advanced Parts (i.e., Bars and Joints)
        self._image_parts = self._apply_and_generate_mask_shapes(
            self._image_parts,
        )

    def compile_masks(self) -> None:
        """
        This method compiles the masks.

        This is a separate method for modularity, specifically for the video
        pipeline.

        """
        # Test Parts for Overlap
        self._image_parts = self._process_state_logic_for_masks(
            self._image_parts,
        )

    def apply_censors(self) -> None:
        """
        This method applies the both the reverse censor and the normal censor.

        Kept separate for stuff like the video pipeline.

        """
        # Generate and Apply Reverse Censor
        self.file_image = self._handle_reverse_censor(
            self.config.reverse_censor.censors,
            Part.create_empty_mask(
                self.file_image.shape,
                inverse=True,
            ),
            self._image_parts,
            self.file_image,
        )

        # Apply Censors
        self.file_image, self.force_png = self._apply_censors(
            self._image_parts,
            self.file_image,
        )

    def start(self) -> None:
        """
        This is the main entrypoint for the ImageProcessor.

        It contains the above public methods and performs the entire image
        pipeline.

        """
        self.generate_parts_and_shapes()
        self.compile_masks()
        self.apply_censors()

    # Util
    def get_part_list(self) -> dict[str, Part]:
        """
        This is a utils method to get the part list.

        :return dict[str, Part]: dictionary of the part names and their part
            object.
        """
        counter = 0
        final_dict = {}
        last_part = ""
        sorted_parts_list = sorted(
            self._image_parts,
            key=lambda part: (part.part_name, part.part_id),
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
        """
        This is a debug function to get the durations from the debugger to
        calculate the total duration of the processor.

        Currently not in use. There used to be "wrappers" that would measure
        the durations but this has been removed since it made the code too
        noisy to read.

        TODO: Reimplement at some point.

        :return float: The total duration value
        """
        return sum([time.duration for time in self._debugger.time_logger])
