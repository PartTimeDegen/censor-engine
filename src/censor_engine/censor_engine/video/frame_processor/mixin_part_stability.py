from .utils import FrameProcessorUtils
from .structs import FramePart


class MixinPartStability(FrameProcessorUtils):
    ...
    # def apply_frame_stability(self) -> None:
    #     """
    #     Applies frame stability by ensuring that parts don't update for small differences.
    #     A significant difference (as determined by frame_difference_threshold) is required to update the part.
    #     """
    #     for part_name in self.current_frame.keys():
    #         # Ignore new parts that don't exist in the last frame
    #         if part_name not in self.last_frame:
    #             continue

    #         old_part = self.last_frame[part_name]
    #         this_part = self.current_frame[part_name]

    #         # Check if the difference between frames is large enough to update
    #         self.current_frame[part_name] = (
    #             this_part if self._compare_frames(old_part, this_part) else old_part
    #         )
