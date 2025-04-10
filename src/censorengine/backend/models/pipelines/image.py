from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import itertools
from typing import Optional

import cv2
import numpy as np
from typing import TYPE_CHECKING

from censorengine.backend.models.tools.debugger import (
    Debugger,
    DebugLevels,
)
from censorengine.backend.models.structures.enums import PartState, ShapeType

if TYPE_CHECKING:
    from censorengine.backend.constants.typing import CVImage
from censorengine.backend.models.structures.detected_part import Part
from censorengine.libs.style_library.catalogue import style_catalogue
from censorengine.backend.models.config import Config

from censorengine.libs.detector_library.catalogue import (
    enabled_detectors,
    enabled_determiners,
)
from censorengine.lib_models.detectors import DetectedPartSchema
from uuid import uuid4, UUID
from censorengine.backend.models.tools.dev_tools import DevTools


@dataclass(slots=True)
class ImageProcessor:
    file_image: "CVImage"
    config: "Config"
    debug_level: DebugLevels = DebugLevels.NONE
    dev_tools: DevTools | None = field(default=None)

    # Internals
    force_png: bool = False

    detected_parts: list[DetectedPartSchema] = field(init=False, default_factory=list)
    extracted_information: dict[str, str] = field(init=False, default_factory=dict)

    parts: list[Part] = field(init=False, default_factory=list)

    empty_mask: "CVImage" = field(init=False)
    file_original_image: "CVImage" = field(init=False)
    file_uuid: UUID = field(init=False)

    debugger: Debugger = field(init=False)
    duration: float = field(init=False)

    def __post_init__(self):
        Part.part_id = itertools.count(start=1)
        self.detected_parts.clear()
        self.extracted_information.clear()
        self.file_uuid = uuid4()
        self.file_original_image = self.file_image

        # Debug
        self.debugger = Debugger(
            "Censor Manager",
            level=self.debug_level,
        )
        self.debugger.time_total_start()
        self.debugger.display_onnx_info()

        # Empty Mask
        self.debugger.time_start("Create Empty Mask")
        self.empty_mask = self._create_empty_mask()
        self.debugger.time_stop()

        # Determine Image
        self.debugger.time_start("Determine Image")
        self.extracted_information = {
            determiner.model_name: determiner.determine_image(self.file_image)
            for determiner in enabled_determiners
        }
        self.debugger.time_stop()

        # Detect Parts for Image
        self.debugger.time_start("Detect Parts")
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
                self.parts if not iter_part else iter_part,
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
                np.ones(
                    self.file_image.shape,
                    dtype=np.uint8,
                )
                * 255
            )

        return Part.normalise_mask(
            np.zeros(
                self.file_image.shape,
                dtype=np.uint8,
            )
        )

    def _create_parts(self):
        """
        This function creates the list of Parts for CensorEngine to keep track
        of. # TODO: Update me to account for split with _detection

        Method:
            1)  It will create an empty list and also find the enabled parts
            2)  It will then collect a list of all of the parts from all of the
                enable AI models/detectors.
            3)  It will then use the parts list to make `Part` objects from the
                found parts, while also discarding any that aren't enabled.
            4)  It then will filter for `None` values (by product of the
                function)

        Notes:
            -   The reason a Map/Filter function is used is because this part
                of the code takes a while to run, using map/filter reduces the
                time massively, as ugly as it looks (I did try to learn it up
                but there's only so much makeup you can put on a pig).

            -   NudeNet (and I assume others) were found to be 98% of the time
                taken for this to run, so it's slow but it's because of the
                package/model itself, not the rest of the code. It's pretty
                much optimised as much as it can be (even the Part creation in
                total was only 0.005s, which is nothing)

        """
        # Acquire Settings
        self.parts.clear()

        # Map and Filter Parts for Missing Information
        def add_parts(detect_part: DetectedPartSchema) -> Optional[Part]:
            """
            Generates the parts using the Part constructor. Also checks that
            the part is in the enabled parts.

            The structure could be improved but the reason I've used a map()
            instead of list comprehensions is because it's faster.

            :param DetectedPartSchema detect_part: Output from the Detector class method
            :return Optional[Part]: A Part object (or None)
            """
            if detect_part.label not in self.config.censor_settings.enabled_parts:
                return

            return Part(
                part_name=detect_part.label,
                score=detect_part.score,
                relative_box=detect_part.relative_box,
                empty_mask=self._create_empty_mask(),
                config=self.config,
                file_uuid=self.file_uuid,
            )

        self.parts = [
            part for part in map(add_parts, self.detected_parts) if part is not None
        ]

    def _merge_parts_if_in_merge_groups(self):
        for index, part in enumerate(self.parts):
            if not part.merge_group:
                continue

            for other_part in self.parts[index + 1 :]:
                if part == other_part:
                    continue
                if other_part.part_name not in part.merge_group:
                    continue

                part.base_masks.append(other_part.base_masks[0])
                self.parts.remove(other_part)

            part.compile_base_masks()

    def _apply_and_generate_mask_shapes(self):
        for part in self.parts:
            # For Simple Shapes
            if not part.is_merged:
                shape_single = Part.get_shape_class(part.shape_object.single_shape)
                part.mask = shape_single.generate(part, self.empty_mask.copy())

            # For Advanced Shapes
            match part.shape_object.shape_type:
                case ShapeType.BASIC:
                    self._decompile_masks("02_advanced_basic", iter_part=part)

                case ShapeType.JOINT:
                    part.mask = part.shape_object.generate(part, self.empty_mask.copy())
                    self._decompile_masks("02_advanced_joint", iter_part=part)

                case ShapeType.BAR:
                    if not part.is_merged:
                        # Make Basic Shape
                        shape_single = Part.get_shape_class("ellipse")
                        part.mask = shape_single.generate(part, self.empty_mask.copy())
                        self._decompile_masks(
                            "02_advanced_bar_01_base_ellipses", iter_part=part
                        )

                    # Make Shape Joint for Bar Basis
                    shape_joint = Part.get_shape_class("joint_ellipse")
                    part.mask = shape_joint.generate(part, self.empty_mask.copy())
                    self._decompile_masks(
                        "02_advanced_bar_02_joint_ellipse", iter_part=part
                    )

                    # Generate Bar
                    part.mask = part.shape_object.generate(part, self.empty_mask.copy())
                    self._decompile_masks("02_advanced_bar_03_bar", iter_part=part)

    def _process_state_logic_for_masks(self):
        sorted_parts = sorted(
            self.parts, key=lambda x: (-x.part_settings.state, x.part_name)[::-1]
        )
        removed_parts = []  # Track removed parts

        for index, primary_part in enumerate(sorted_parts):
            if primary_part in removed_parts:
                continue

            primary_state = primary_part.part_settings.state

            for secondary_part in sorted_parts[index + 1 :]:
                if secondary_part in removed_parts:
                    continue

                secondary_state = secondary_part.part_settings.state

                # Quality of Life Booleans
                same_censors = (
                    primary_part.part_settings.censors
                    == secondary_part.part_settings.censors
                )
                same_state = primary_state == secondary_state
                primary_has_higher_rank = primary_state > secondary_state

                def subtract_masks(target: Part, source: Part):
                    """Subtract one part’s mask from another."""
                    target.subtract(source.mask)

                def combine_parts():
                    """Combine secondary part into primary and remove secondary."""
                    primary_part.add(secondary_part.mask)
                    removed_parts.append(secondary_part)
                    self.parts.remove(secondary_part)

                # MATCHING: Merge if both parts have the same settings
                if same_censors and same_state:
                    combine_parts()

                # PROTECTED: If `censors` match, combine instead of subtracting
                elif (
                    primary_state == PartState.PROTECTED
                    or secondary_state == PartState.PROTECTED
                ):
                    if same_censors:
                        combine_parts()
                    elif same_state or primary_state == PartState.PROTECTED:
                        subtract_masks(secondary_part, primary_part)
                    else:
                        subtract_masks(primary_part, secondary_part)

                # REVEALED: Higher-ranked part subtracts from lower-ranked part
                elif primary_state == PartState.REVEALED:
                    subtract_masks(
                        secondary_part if primary_has_higher_rank else primary_part,
                        primary_part if primary_has_higher_rank else secondary_part,
                    )

                # UNPROTECTED: Merge if same censors, otherwise subtract
                elif primary_state == PartState.UNPROTECTED:
                    if primary_has_higher_rank and same_censors:
                        secondary_part.add(primary_part.mask)
                        removed_parts.append(primary_part)
                        self.parts.remove(primary_part)
                    elif primary_has_higher_rank:
                        subtract_masks(primary_part, secondary_part)

    def _handle_reverse_censor(self):
        if not self.config.reverse_censor.censors:
            return

        # Create Mask
        base_mask_reverse = self._create_empty_mask(inverse=True)
        for part in self.parts:
            base_mask_reverse = cv2.subtract(base_mask_reverse, part.mask)

        # Apply Censors
        contour = cv2.findContours(
            image=base_mask_reverse,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        for censor in self.config.reverse_censor.censors[::-1]:
            censor_object = style_catalogue[censor.function]()
            censor_object.change_linetype(enable_aa=False)
            censor_object.using_reverse_censor = True

            arguments = [
                self.file_image,
                contour,
            ]

            self.file_image = censor_object.apply_style(
                *arguments,
                **censor.args,
            )

    def _apply_censors(self):
        parts = sorted(self.parts, key=lambda x: (x.part_settings.state, x.part_name))

        working_image = self.file_image.copy()
        for part in parts:
            if not part.part_settings.censors or not part:
                continue

            # if part.use_global_area:
            part_contour = cv2.findContours(
                image=part.mask,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )  # TODO: reduce image size to just part

            # Gather default args
            arguments = {
                "image": working_image,
                "contour": part_contour,
            }

            # Reversed to represent YAML order
            for censor in part.part_settings.censors[::-1]:
                # Acquire Function
                censor_object = style_catalogue[censor.function]()

                # Turn on AA
                censor_object.change_linetype(enable_aa=True)

                # Handle Args
                if censor_object.style_type == "dev":
                    arguments["part"] = part
                elif arguments.get("part"):
                    arguments.pop("part")

                # Apply Censor
                arguments["image"] = censor_object.apply_style(
                    **arguments,
                    **censor.args,
                )

                # Forces PNG if the Censor Requires it
                if censor_object.force_png:
                    self.force_png = censor_object.force_png

            # Apply Potential Feather Fade
            if not part.part_settings.fade_percent:
                working_image = arguments["image"]
                continue

            # # Get Mask
            contour_mask = cv2.drawContours(
                np.zeros(self.file_image.shape, dtype=np.uint8),
                part_contour[0],
                -1,
                (255, 255, 255),
                -1,
            )

            # # Convert Mask to Right Format
            contour_mask = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY)
            contour_mask = contour_mask.astype(float) / 255.0

            # # Calculate Feather Effect based on Contour
            _, _, w, h = cv2.boundingRect(part_contour[0][0])
            fade_percent = np.clip(int(part.part_settings.fade_percent), 0, 100)
            max_dim = max(w, h)
            feathering_amount = int((fade_percent / 100.0) * max_dim)
            kernel_size = min(51, max(3, feathering_amount // 2 * 2 + 1))

            # # Apply Gaussian blur for feathering
            contour_mask = cv2.erode(
                contour_mask,
                np.ones((kernel_size, kernel_size), np.uint8),
                iterations=1,
            ).astype(float)
            feathered_mask = cv2.GaussianBlur(
                contour_mask, (kernel_size, kernel_size), 0
            )

            # # Convert the mask back to 3-channel for blending
            feathered_mask = cv2.merge([feathered_mask] * 3)  # type: ignore

            # # Blend the images using the feathered mask
            feathered_mask *= 1.0
            merged_image = (
                feathered_mask * working_image + (1 - feathered_mask) * self.file_image
            )

            working_image = merged_image

        self.file_image = working_image

    # Public
    def return_output(self):
        return self.file_image

    def generate_parts_and_shapes(self):
        # Create Parts
        self._decompile_masks("00_stage_base_00_create_part")
        self.debugger.time_start("Create Parts")
        self._create_parts()
        self.debugger.time_stop()
        self._decompile_masks("00_stage_result_00_create_part")

        # Merge Parts
        self._decompile_masks("00_stage_base_01_merged")
        self.debugger.time_start("Merge Parts")
        self._merge_parts_if_in_merge_groups()
        self.debugger.time_stop()
        self._decompile_masks("00_stage_result_01_merged")

        # Handle More Advanced Parts (i.e., Bars and Joints)
        self._decompile_masks("00_stage_base_02_advanced")
        self.debugger.time_start("Handle Advanced Shapes")
        self._apply_and_generate_mask_shapes()
        self.debugger.time_stop()
        self._decompile_masks("00_stage_result_02_advanced")

    def compile_masks(self):
        # Test Parts for Overlap
        self.debugger.time_start("Process State Logic")
        self._process_state_logic_for_masks()
        self.debugger.time_stop()

    def apply_censors(self):
        # Generate and Apply Reverse Censor
        self.debugger.time_start("Generate Reverse Censor")
        self._handle_reverse_censor()
        self.debugger.time_stop()

        # Apply Censors
        self.debugger.time_start("Apply Censors")
        self._apply_censors()
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
            self.parts, key=lambda part: (part.part_name, part.part_id)
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
