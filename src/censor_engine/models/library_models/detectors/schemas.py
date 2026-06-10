import numpy as np
from pydantic import BaseModel, field_validator


class DetectedPart(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            np.ndarray: lambda v: v.tolist(),
        },
    }

    origin: str

    # Internal
    part_id: int = 0

    # Meta
    label: str | None = None
    score: float | None = None

    # Data Used for Information
    bbox: tuple[int, int, int, int] | None = None  # XYXY
    masks: list[np.ndarray] | None = None

    def set_part_id(self, number: int) -> None:
        self.part_id = number

    @field_validator("bbox", mode="before")
    @classmethod
    def convert_bbox(cls, v):  # noqa: ANN001, ANN206
        if v is None:
            return None

        return tuple(round(float(x)) for x in v)
