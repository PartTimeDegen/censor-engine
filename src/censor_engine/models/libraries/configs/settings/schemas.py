from pydantic import BaseModel

from censor_engine.models.libraries.configs._helper_types import (
    MarginPercentage,
)


class Margins(BaseModel):
    height: MarginPercentage = 0.0
    width: MarginPercentage = 0.0
