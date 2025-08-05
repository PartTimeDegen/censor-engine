import uuid

from censor_engine.models.structs import Mixin
from .structs import FramePart
from typing import Iterable


class FrameProcessorUtils(Mixin):
    def load_parts_from_frame(
        self, list_of_frameparts: Iterable[FramePart]
    ) -> dict[str, FramePart]:
        return {str(uuid.uuid4()): frame_part for frame_part in list_of_frameparts}
