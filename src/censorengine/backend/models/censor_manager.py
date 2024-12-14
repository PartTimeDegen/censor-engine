from dataclasses import dataclass, field
import itertools
import statistics
import timeit
from typing import Optional

import cv2
import numpy as np
from typing import TYPE_CHECKING

from censorengine.backend.models.enums import PartState, ShapeType

if TYPE_CHECKING:
    from censorengine.backend.constants.typing import (
        CVImage,
        Config,
        NudeNetInfo,
    )
from censorengine.backend.models.detected_part import Part
from censorengine.libs.style_library.catalogue import style_catalogue

from nudenet import NudeDetector  # type: ignore


@dataclass
class CensorManager:
    # Parts
    parts: list[Optional[Part]] = field(default_factory=list)

    # Common Masks
    empty_mask: "CVImage" = field(init=False)

    # File Info
    file_original_image: "CVImage" = field(init=False)
    file_image: "CVImage" = field(init=False)
    file_loc: str = field(init=False)
    file_image_name: str = field(init=False)
    force_png: bool = False

    # TODO: Move to Debugger
    debug_level: int = field(default=0)
    debug_time_logger: list[tuple[int, str, float, float]] = field(
        default_factory=list[tuple[int, str, float, float]]
    )

    # Manager Info
    config: "Config" = field(init=False)

    # Stats
    # TODO: Expand on part_type_id, make an Enum and can include part "area"
    # (ie group exposed and covered variants)
    # part_type_id: int
    # part_type_counts: int

    def __init__(
        self,
        file_path: str,
        config: "Config",
        show_duration: bool = False,
        debug_level: int = 0,
        debug_log_time: bool = False,
        index_text: str = "",
    ):
        self.config = config

        # Statistics
        timer_start = timeit.default_timer()
        self.debug_time_logger = [(1, "init", timer_start, 0.0)]

        # Debug
        self.debug_level = debug_level
        Part.part_id = itertools.count(start=1)

        # File Stuff
        self.file_loc = file_path
        self.file_image_name = file_path.split("/")[-1]

        # NOTE: This may produce
        #       "libpng warning: iCCP: known incorrect sRGB profile" errors
        #       I tried suppressing them but it doesn't work
        self.file_original_image = cv2.imread(file_path)
        self.file_image = cv2.imread(file_path)

        # Declare Start
        if index_text != "":
            index_text += " "
        print()
        print(f'{index_text}Censoring: "{self.file_image_name}"')

        # Empty Mask
        self.empty_mask = self._create_empty_mask()

        # NudeNet Stuff
        self._append_parts()

        # Merge Parts
        self._merge_parts_if_in_merge_groups()

        # Handle More Advanced Parts (i.e., Bars and Joints)
        self._apply_mask_shapes()

        # Test Parts for Overlap
        self._process_overlaps_for_masks()

        # Generate and Apply Reverse Censor
        self._handle_reverse_censor()

        # Apply Censors
        self._apply_censors()

        # Print Duration
        timer_stop = timeit.default_timer()
        duration = timer_stop - timer_start
        self.stats_duration = duration

        # Display Output
        self.display()
        if show_duration:
            print(f"- Duration:\t{self.stats_duration:0.3f} seconds")
        print()

    def display(self):
        count = 1
        if self.debug_level == 0:
            return

        print("- Parts Found:")  # TODO: Move this function to Debug Class
        for part in self.parts:
            print(f"- {count:02d}) {part.part_name}")
            count += 1

            if self.debug_level >= 1:
                print(f"- - Score             : {part.score:02.0%}")
                print(f"- - Box               : {part.box}")
                print(f"- - Level             : {part.state}")
                print(f"- - Merge Group       : {part.merge_group}")
                print(f"- - Shape             : {part.shape.shape_name}")

            if self.debug_level >= 2:
                print(f"- - ID                : {part.part_id}")
                print(f"- - Merge ID          : {part.merge_group_id}")
                print(f"- - Censors           : {part.censors}")
                print(f"- - Protected shape   : {part.protected_shape}")

    # Private
    def _create_empty_mask(self, inverse: bool = False):
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

    def _append_parts(self):
        self.parts = []
        config_parts_enabled = self.config.parts_enabled
        detected_parts = NudeDetector().detect(self.file_loc)

        def add_parts(detect_part: "NudeNetInfo"):
            if detect_part["class"] not in config_parts_enabled:
                return

            return Part(
                nude_net_info=detect_part,
                empty_mask=self._create_empty_mask(),
                config=self.config,
                file_path=self.file_loc,
            )

        self.parts = list(map(add_parts, detected_parts))
        self.parts = list(filter(lambda x: x is not None, self.parts))

    def _merge_parts_if_in_merge_groups(self):
        if not self.config.merge_enabled:
            return

        merge_groups = self.config.merge_groups
        if not merge_groups:
            return

        full_parts = self.parts
        for index, part in enumerate(full_parts):
            if not part.merge_group:
                continue

            for other_part in full_parts[index + 1 :]:
                if part == other_part:
                    continue
                if other_part.part_name not in part.merge_group:
                    continue

                part.base_masks.append(other_part.base_masks[0])
                self.parts.remove(other_part)

            part.compile_base_masks()

    def _apply_mask_shapes(self):
        for part in self.parts:
            # For Single Shapes
            if not part.is_merged:
                shape_single = Part.get_shape_class(part.shape.single_shape)

                if not shape_single:
                    raise KeyError("Missing single shape")

                part.mask = shape_single.generate(part, self.empty_mask.copy())
                continue

            # For Advanced Shaoes
            match part.shape.shape_type:
                case ShapeType.BASIC:
                    pass

                case ShapeType.JOINT:
                    part.mask = part.shape.generate(
                        part, self.empty_mask.copy()
                    )

                case ShapeType.BAR:
                    shape_joint = Part.get_shape_class("joint_ellipse")
                    part.mask = shape_joint.generate(
                        part, self.empty_mask.copy()
                    )
                    part.mask = part.shape.generate(
                        part, self.empty_mask.copy()
                    )

    def _process_overlaps_for_masks(self):
        full_parts = sorted(self.parts, key=lambda x: (-x.state, x.part_name))[
            ::-1
        ]

        for index, target_part in enumerate(full_parts):
            if target_part not in self.parts:
                continue

            for comp_part in full_parts[index + 1 :]:
                if (
                    target_part not in self.parts
                    or comp_part not in self.parts
                ):
                    continue
                # NOTE: This is done in respect to Target

                # Quality of Life Booleans
                compared_ettings = {
                    "censors": target_part.censors == comp_part.censors,
                    "states": target_part.state == comp_part.state,
                }

                is_comp_higher = target_part.state <= comp_part.state

                # Flow Chart
                # # MATCHING
                if all(compared_ettings.values()):
                    # ALL MATCHING CRITERIA
                    # Combine Parts
                    target_part.add(comp_part.mask)
                    self.parts.remove(comp_part)

                # # PROTECTED
                elif (
                    target_part.state == PartState.PROTECTED
                    or comp_part.state == PartState.PROTECTED
                ):
                    if compared_ettings["states"]:
                        comp_part.subtract(target_part.mask)
                    elif target_part.state == PartState.PROTECTED:
                        comp_part.subtract(target_part.mask)
                    elif comp_part.state == PartState.PROTECTED:
                        target_part.subtract(comp_part.mask)

                # # REVEALED
                elif target_part.state == PartState.REVEALED:
                    if is_comp_higher:
                        comp_part.subtract(target_part.mask)
                    else:
                        target_part.subtract(comp_part.mask)

                # # UNPROTECTED
                elif target_part.state == PartState.UNPROTECTED:
                    if is_comp_higher and compared_ettings["censors"]:
                        comp_part.add(target_part.mask)
                        self.parts.remove(target_part)

                    elif is_comp_higher:
                        target_part.subtract(comp_part.mask)

    def _handle_reverse_censor(self):
        if not self.config.reverse_censor_enabled:
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

        for censor in self.config.reverse_censors[::-1]:
            censor_object = style_catalogue[censor.function]()
            censor_object.change_linetype(enable_aa=False)
            censor_object.using_reverse_censor = True

            arguments = [
                self.file_image,
                contour,
            ]
            if censor_object.style_type == "dev":
                arguments.append(part)

            self.file_image = censor_object.apply_style(
                *arguments,
                **censor.args,
            )

    def _apply_censors(self):
        parts = sorted(self.parts, key=lambda x: (x.state, x.part_name))

        for part in parts:
            if not part.censors:
                continue

            part_contour = cv2.findContours(
                image=part.mask,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )

            # Reversed to represent YAML order
            for censor in part.censors[::-1]:
                censor_object = style_catalogue[censor.function]()
                censor_object.change_linetype(enable_aa=True)
                arguments = [
                    self.file_image,
                    part_contour,
                ]
                if censor_object.style_type == "dev":
                    arguments.append(part)

                self.file_image = censor_object.apply_style(
                    *arguments,
                    **censor.args,
                )

    # Lists of Parts
    def get_list_of_parts_total(self, search: Optional[dict[str, str]] = None):
        if not search:
            return self.parts

        list_matching_parts_attributes = [
            part
            for part in self.parts
            if all(
                part.__dict__.get(key)
                and (str(part.__dict__.get(key)) == str(value))
                for key, value in search.items()
            )
        ]

        return list_matching_parts_attributes

    # Dev
    def logging_decompose_mask(self, part_name: str, prefix: str):
        pass

    # Static
    @staticmethod
    def get_statistics(
        durations: list[float],
    ) -> dict[str, float]:
        collection = {}
        collection["mean"] = statistics.mean(durations)
        collection["median"] = statistics.median(durations)
        collection["max"] = max(durations)
        collection["min"] = min(durations)
        collection["range"] = max(durations) - min(durations)

        if len(durations) != 1:
            collection["stdev"] = statistics.stdev(durations)
            if collection["stdev"] and collection["mean"] != 0.0:
                collection["coefficient_of_variation"] = (
                    collection["stdev"] / collection["mean"]
                )

        return collection

    def return_output(self):
        return self.file_image
