from dataclasses import dataclass, field
from typing import Iterable
from .structs import FramePart
from .mixin_part_persistence import MixinPartPersistence

from censor_engine.detected_part import Part


@dataclass(slots=True)
class FrameProcessor(MixinPartPersistence):
    """
    This class handles the processing of the parts between frames, such to
    improve the quality of the output.

    Currently, # TODO
    # FIXME Make persistence work with parts rather than frame (see name change)

    """

    frame_difference_threshold: float  # Minimum Required difference to change
    part_frame_hold_frames: int

    frame_lag_counter: int = field(default=0, init=False)

    loaded_frame: dict[str, FramePart] = field(default_factory=dict, init=False)
    current_frame: dict[str, FramePart] = field(default_factory=dict, init=False)

    first_frame: bool = field(default=True, init=False)

    part_dictionary: dict[int, FramePart] = field(default_factory=dict, init=False)

    def load_parts(self, parts: list[Part]) -> None:
        # TODO Too tired to do now, but an approximate area of the box might be worth it, then having a double forloop to go through and find new ones, can optimise after
        frame_parts = [FramePart(part) for part in parts]
        self.current_frame = self.load_parts_from_frame(frame_parts)

        if self.first_frame:
            self.first_frame = False

        # Save for Debugging
        self.loaded_frame = self.current_frame.copy()

        self.part_dictionary = self.determine_frame_contingency(
            self.part_dictionary,
            self.current_frame,
        )

    def run(self):
        # Persistence
        self.current_frame, self.part_dictionary = self.apply_part_persistence(
            self.current_frame,
            self.part_dictionary,
            self.part_frame_hold_frames,
        )

    def retrieve_parts(self) -> list[Part]:
        return [part.part for part in self.current_frame.values()]

    @staticmethod
    def get_list_of_frame_parts(list_of_parts: Iterable[FramePart]) -> list[str]:
        return [part.get_debug_text() for part in list_of_parts]
