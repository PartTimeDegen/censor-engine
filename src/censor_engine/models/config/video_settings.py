from dataclasses import dataclass, field


@dataclass(slots=True)
class VideoConfig:
    # Core Settings
    # NOTE: The default "-1" means to use the native FPS
    censoring_fps: int = -1  # TODO
    output_fps: int = -1  # TODO

    # Video Cleaning Settings
    # # Frame Stability Config
    frame_difference_threshold: float = 0.05

    # Frame Part Persistence Config
    part_frame_hold_seconds: float = -1.0
    persistence_groups: list[list[str]] = field(default_factory=list)

    def __post_init__(self):
        # Type Narrow to Float
        if isinstance(self.frame_difference_threshold, int):
            self.frame_difference_threshold = float(
                self.frame_difference_threshold
            )
        if isinstance(self.part_frame_hold_seconds, int):
            self.part_frame_hold_seconds = float(self.part_frame_hold_seconds)
