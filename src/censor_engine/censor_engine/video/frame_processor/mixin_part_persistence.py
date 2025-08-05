from .utils import FrameProcessorUtils
from .structs import FramePart


class MixinPartPersistence(FrameProcessorUtils):
    def _find_part_index_in_dictionary(
        self,
        part_dictionary: dict[int, FramePart],
        frame_value: FramePart,
    ) -> int | None:
        part_candidates = []
        for index, dict_value in part_dictionary.items():
            # Conditions
            is_matching_part_type_merged = (
                dict_value.part.get_name_and_merged()
                == frame_value.part.get_name_and_merged()
            )

            is_matching_merge = dict_value.part.is_merged == frame_value.part.is_merged

            if group := frame_value.part.persistence_group_id:
                is_matching_persistence_group = (
                    dict_value.part.persistence_group_id == group
                )
            else:
                is_matching_persistence_group = False

            is_within_approx_region = dict_value.part.part_area.check_in_approx_region(
                frame_value.part.part_area.region
            )

            # Save Index to List
            is_full_change = is_matching_merge and is_matching_persistence_group
            is_partial_change = not is_matching_merge and (
                is_within_approx_region and is_matching_persistence_group
            )
            is_part = (
                is_matching_part_type_merged or is_full_change or is_partial_change
            )

            if is_part:
                part_candidates.append(index)

        # Return Youngest Part
        if part_candidates:
            selected_index = min(
                part_candidates,
                key=lambda index: part_dictionary[index].lifespan_frames,
            )
            return selected_index

        return None

    def determine_frame_contingency(
        self,
        part_dictionary: dict[int, FramePart],
        current_frame: dict[str, FramePart],
    ) -> dict[int, FramePart]:
        # Find Starting Point
        if len(part_dictionary) != 0:
            max_index = max(part_dictionary.keys())
        else:
            max_index = 0

        # Check Values
        for value in current_frame.values():
            index = self._find_part_index_in_dictionary(part_dictionary, value)

            if index:
                part_dictionary[index] = value
            else:
                max_index += 1
                part_dictionary[max_index] = value

        return part_dictionary

    def apply_part_persistence(
        self,
        current_frame: dict[str, FramePart],
        part_dictionary: dict[int, FramePart],
        frame_hold_limit: int,
    ) -> tuple[dict[str, FramePart], dict[int, FramePart]]:
        """
        Holds the frame for the duration specified by frame_hold_amount.
        If the frame has been held for enough time, it will be updated.

        """

        # Remove Parts with Lifespan too Old else Age
        for key, value in list(part_dictionary.items()):
            value.lifespan_frames += 1
            if value.lifespan_frames > frame_hold_limit:
                part_dictionary.pop(key)

        # Apply the Parts
        current_frame = self.load_parts_from_frame(part_dictionary.values())
        return current_frame, part_dictionary
