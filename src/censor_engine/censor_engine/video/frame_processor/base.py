from dataclasses import dataclass, field

from censor_engine.censor_engine.video.frame_processor.structs import Tracker


@dataclass(slots=True)
class FrameProcessor:
    """
    This class handles the processing of the parts between frames, such to
    improve the quality of the output.

    # TODO: Continue
    # FIXME:  Make persistence work with parts rather than frame
    # (see name change)

    """

    maximum_miss_frame: int

    frame_lag_counter: int = field(default=0, init=False)

    tracker: Tracker = field(init=False)

    def __post_init__(self):
        self.tracker = Tracker(max_missed=self.maximum_miss_frame)
