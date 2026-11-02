import numpy as np
from pydantic import BaseModel, NonNegativeInt


class DepthProperties(BaseModel):
    # Config
    model_config = {"arbitrary_types_allowed": True}

    # Layering
    centroid_depth: np.float32
    censor_layer: NonNegativeInt


class AdditionalProperties(BaseModel):
    depth: DepthProperties | None = None
