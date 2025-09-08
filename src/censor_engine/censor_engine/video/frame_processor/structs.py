from dataclasses import dataclass, field

from censor_engine.detected_part import Part


@dataclass(slots=True)
class FramePart:
    part: Part
    lifespan_frames: int = 0

    part_name: str = field(init=False)
    is_merged: bool = field(init=False)

    def __post_init__(self):
        self.part_name = self.part.get_name_and_merged()
        self.is_merged = self.part.is_merged

    def __repr__(self):
        # return f"{self.part_name}"
        return f"{self.part_name}_lifespan={self.lifespan_frames}"

    def get_debug_text(self) -> str:
        return f"{self.part.get_id_name_and_merged()} ({self.part_name})"
