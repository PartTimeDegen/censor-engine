from dataclasses import dataclass, field
from uuid import UUID, uuid4

from censor_engine.core.paths import PathManager
from censor_engine.core_engine.tools.debugger import (
    DebugLevel,
)
from censor_engine.core_engine.tools.dev_tools import DevTools
from censor_engine.models.caching import Cache
from censor_engine.models.config import Config
from censor_engine.models.detected_part import Part
from censor_engine.models.lib_models.detectors.ai_models import AIModel
from censor_engine.models.lib_models.detectors.schemas import (
    DetectedPart,
)

from censor_engine._typing import Image

from .mixin_compile_masks import MixinComponentCompile
from .mixin_detect_parts import MixinDetectParts
from .mixin_generate_censors import MixinGenerateCensors
from .mixin_generate_parts import MixinGenerateParts


@dataclass(slots=True)
class ImageProcessor(
    MixinDetectParts,
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
    file_name: str
    path_manager: PathManager

    config: Config
    live_detectors: list[AIModel]

    cache: Cache | None
    frame_counter: int | None = None

    debug_level: DebugLevel = DebugLevel.NONE
    dev_tools: DevTools | None = None

    _test_detection_output: list[DetectedPart] | None = None

    # Internals
    force_png: bool = False

    _detected_parts: list[DetectedPart] = field(
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

    _duration: float = field(init=False)

    def __post_init__(self):

        self._file_original_image = self.file_image.copy()

        # Detect Parts for Image
        if self._test_detection_output:
            self._detected_parts = self._test_detection_output
        else:
            self._detected_parts = self._detect_parts(
                frame=self.frame_counter,
                cache=self.cache,
                detectors=self.live_detectors,
                image=self.file_image,
            )

    # Post-Init Helpers
    def __clear_per_frame_memory(self):
        self._detected_parts.clear()
        self._extracted_information.clear()
        self._file_uuid = uuid4()

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

    def generate_parts(self) -> None:
        """
        This method handles the generation of the parts.

        Stages:
            1)  Create parts.
            2)  Filter parts that don't meet the minimum score threshold.

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

    def generate_mask_masks(self) -> None:
        """
        This method handles the generation the masks' masks.

        Stages:
            1)  Merge parts based on the merge method and merge groups.
            2)  Apple the mask effects to the mask, handling more advanced
                parts as well which require more than one pass.

        """
        # Merge Parts
        self._image_parts = self._merge_parts(
            self._image_parts,
        )

        # Handle More Advanced Parts (i.e., Bars and Joints)
        self._image_parts = self._apply_and_generate_mask_masks(
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
        self.generate_parts()
        self.generate_mask_masks()
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
