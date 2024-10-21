from dataclasses import dataclass, field
import itertools
import statistics
import timeit
from typing import Optional

import cv2
import numpy as np

from censorengine.backend.models.enums import PartLevel
from censorengine.backend.constants.typing import CVImage, Config
from censorengine.backend.models.detected_part import Part
from censorengine.libs.style_library.catalogue import style_catalogue

from nudenet import NudeDetector  # type: ignore


@dataclass
class CensorManager:
    # Parts
    parts: list[Optional[Part]] = field(default_factory=list)

    # Common Masks
    # # Reverse
    reverse_censor_mask: Optional[CVImage] = None
    empty_mask: CVImage = field(init=False)

    # File Info
    file_original_image: CVImage = field(init=False)
    file_image: CVImage = field(init=False)
    file_loc: str = field(init=False)
    file_image_name: str = field(init=False)
    force_png: bool = False

    debug_level: int = field(default=0)
    debug_time_logger: float = field(default_factory=list[tuple[str, float]])
    stats_duration: float = field(init=False)

    # Manager Info
    config: Config = field(default=0)

    # Stats
    # TODO: Expand on part_type_id, make an Enum and can include part "area"
    # (ie group exposed and covered variants)
    # part_type_id: int
    # part_type_counts: int

    def __init__(
        self,
        file_path: str,
        config: Config,
        show_duration: bool = False,
        debug_level: int = 0,
        debug_log_time: bool = False,
    ):
        self.config = config

        # Statistics
        timer_start = timeit.default_timer()
        self.debug_time_logger = [(1, "init", timer_start, 0.0)]

        # Debug
        self.debug_level = debug_level
        Part.part_id = itertools.count(start=1)
        self._debug_save_time("debug")

        # File Stuff
        self.file_loc = file_path
        self.file_image_name = file_path.split("/")[-1]
        self.file_original_image = cv2.imread(file_path)
        self.file_image = cv2.imread(file_path)
        self._debug_save_time("save_info")

        # Declare Start
        print()
        print(f'Censoring: "{self.file_image_name}"')

        # Empty Mask
        self._create_empty_mask()
        self._debug_save_time("create_mask")

        # NudeNet Stuff
        self._append_parts()
        self._debug_save_time("add_parts")

        # Merge Parts
        self._merge_parts_if_in_merge_groups()
        self._debug_save_time("merge_parts")

        # Test Parts for Overlap
        self._process_overlaps_for_masks()
        self._debug_save_time("process_overlap_for_parts")

        # Apply Censors
        self._apply_censors()
        self._debug_save_time("apply_censors")

        # Save File
        cv2.imwrite("zzz_test/" + self.file_image_name, self.file_image)

        # Print Duration
        timer_stop = timeit.default_timer()
        duration = timer_stop - timer_start
        self.stats_duration = duration

        # Display Output
        self.display()
        if show_duration:
            print(f"- Duration:\t{self.stats_duration:0.3f} seconds")

        if debug_log_time:
            self._display_debug_times()
        print()

    def display(self):
        count = 1
        if self.debug_level == 0:
            return

        print("- Parts Found:")
        for part in self.parts:
            print(f"- {count:02d}) {part.part_name}")
            count += 1

            if self.debug_level >= 1:
                print(f"- - Score             : {part.score:02.0%}")
                print(f"- - Box               : {part.box}")
                print(f"- - Level             : {part.part_level}")
                print(f"- - Merge Group       : {part.merge_group}")
                print(f"- - Shape             : {part.shape.shape_name}")

            if self.debug_level >= 2:
                print(f"- - ID                : {part.part_id}")
                print(f"- - Merge ID          : {part.merge_group_id}")
                print(f"- - Censors           : {part.censors}")
                print(f"- - Protected shape   : {part.protected_shape}")

            if self.debug_level >= 3:
                print(f"- - Default censor    : {part.is_default_censors}")
                print(f"- - Default shape     : {part.is_default_shape}")

    # Private
    def _debug_save_time(self, name):
        new_id = self.debug_time_logger[-1][0] + 1
        self.debug_time_logger.append(
            (
                new_id,
                name,
                timeit.default_timer(),
                timeit.default_timer() - self.debug_time_logger[-1][2],
            )
        )

    def _display_debug_times(self):
        durations = [time[-1] for time in self.debug_time_logger]
        min_duration = min([time for time in durations if time > 0.001])
        if min_duration > 0.001:
            times = [
                list(time) + [time[-1] / min_duration]
                for time in self.debug_time_logger
            ]
        else:
            times = [list(time) + [""] for time in self.debug_time_logger]

        times = reversed(sorted(times, key=lambda x: x[3]))

        print("")
        print("=== Debug Durations ===")
        for time_id, time_name, _, duration, relative in times:
            if time_name == "init":
                continue

            spacing = 40
            gap = " " * (spacing - len(time_name))
            addtional_text = (
                f", {relative:2.1f}x minimum" if min_duration > 0.001 else ""
            )
            print(
                f"{time_id:>2} {time_name}{gap}: {duration:0.3f} seconds{addtional_text}"
            )

    def _create_empty_mask(self):
        return Part.normalise_mask(
            np.zeros(
                self.file_image.shape,
                dtype=np.uint8,
            )
        )

    def _append_parts(self):
        self.parts = []
        config_parts_enabled = self.config["parts_enabled"]
        detected_parts = NudeDetector().detect(self.file_loc)

        def add_parts(detect_part: list[dict]):
            if detect_part["class"] not in config_parts_enabled:
                return
            return Part(
                nude_net_info=detect_part,
                empty_mask=self._create_empty_mask(),
                config=self.config,
            )

        self.parts = list(map(add_parts, detected_parts))
        self.parts = list(filter(lambda x: x is not None, self.parts))

    def _merge_parts_if_in_merge_groups(self):
        config_info = self.config["information"]

        config_merge_info = config_info.get("merging")
        if not config_merge_info:
            return

        merge_groups = config_merge_info.get("merge_groups")
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

    def _process_overlaps_for_masks(self):
        full_parts = sorted(
            self.parts, key=lambda x: (x.part_level, x.part_name)
        )[::-1]

        for index, target_part in enumerate(full_parts):
            if target_part not in self.parts:
                continue

            for comp_part in full_parts[index + 1 :]:
                if target_part not in self.parts:
                    continue
                if comp_part not in self.parts:
                    continue
                # NOTE: This is done in respect to Target

                # Quality of Life Booleans
                is_matching_censors = target_part.censors == comp_part.censors
                is_equal_levels = (
                    target_part.part_level == comp_part.part_level
                )
                is_comp_higher = target_part.part_level < comp_part.part_level
                is_targ_higher = target_part.part_level >= comp_part.part_level

                # Flow Chart
                if target_part.part_level == PartLevel.PROTECTED:
                    # TARGET == PROTECTED
                    # Equal levels and censors = Merge into Target
                    # Equal or less levels = Target gets priority

                    if is_equal_levels and is_matching_censors:
                        target_part.add(comp_part.mask)
                        self.parts.remove(comp_part)

                    elif is_targ_higher:
                        comp_part.subtract(target_part.mask)

                elif target_part.part_level == PartLevel.REVEALED:
                    # TARGET == REVEALED
                    # Comp is protected = Ignore
                    # Otherwise = subtract target from comp

                    if comp_part.part_level != PartLevel.PROTECTED:
                        comp_part.subtract(target_part.mask)

                elif target_part.part_level < PartLevel.REVEALED:
                    # TARGET < PROTECTED
                    # Comp higher level and equal censores = Merge into comp
                    # Comp higher level = Comp gets priority
                    # Equal levels and censors = Merge into Target
                    # Equal or less levels = Target gets priority

                    if is_comp_higher and is_matching_censors:
                        comp_part.add(target_part.mask)
                        self.parts.remove(target_part)

                    elif is_comp_higher:
                        target_part.subtract(comp_part.mask)

                    elif is_equal_levels and is_matching_censors:
                        target_part.add(comp_part.mask)
                        self.parts.remove(comp_part)

                    elif is_targ_higher:
                        comp_part.subtract(target_part.mask)

    def _apply_censors(self):
        parts = sorted(self.parts, key=lambda x: (x.part_level, x.part_name))

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
                self.file_image = style_catalogue[censor["function"]](
                    self.file_image,
                    part_contour,
                    **censor["args"],
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
    def get_statistics(durations: list[float]):
        collection = {}
        collection["mean"] = statistics.mean(durations)
        collection["median"] = statistics.median(durations)
        collection["max"] = max(durations)
        collection["min"] = min(durations)
        collection["range"] = max(durations) - min(durations)

        if len(durations) != 1:
            collection["stdev"] = statistics.stdev(durations)

            collection["coefficient_of_variation"] = (
                collection["stdev"] / collection["mean"]
                if collection["mean"] != 0.0
                else None
            )
            collection["quartiles"] = statistics.quantiles(durations)

        return collection

    def return_output(self):
        return self.file_image
