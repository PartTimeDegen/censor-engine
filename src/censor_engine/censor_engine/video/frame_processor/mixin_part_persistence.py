from .utils import FrameProcessorUtils
from .structs import FramePart


class MixinPartPersistence(FrameProcessorUtils):
    def _find_part_in_dictionary(
        self,
        part_dictionary: dict[int, FramePart],
        frame_value: FramePart,
    ) -> int | None:
        for index, dictionary_value in part_dictionary.items():
            is_matching_part_type = (
                dictionary_value.part.get_name() == frame_value.part.get_name()
            )
            is_within_approx_region = (
                dictionary_value.part.part_area.check_in_approx_region(
                    frame_value.part.part_area.region
                )
            )
            if is_matching_part_type and is_within_approx_region:
                return index

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
            if index := self._find_part_in_dictionary(part_dictionary, value):
                part_dictionary[index] = value
            else:
                max_index += 1
                part_dictionary[max_index] = value
        print(part_dictionary)
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

        Methodology:


        :param dict[str, FramePart] current_frame: _description_
        :param dict[str, FramePart] held_frame: _description_
        :param int frame_hold_limit: _description_

        :return dict[str, FramePart]: _description_

        """

        # Remove Parts with Lifespan too Old else Age
        for key, value in list(part_dictionary.items()):
            value.lifespan_frames += 1
            if value.lifespan_frames > frame_hold_limit:
                part_dictionary.pop(key)

        # Apply the Parts
        current_frame = self.load_parts_from_frame(part_dictionary.values())
        return current_frame, part_dictionary
