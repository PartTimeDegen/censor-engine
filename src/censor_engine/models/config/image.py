from pydantic import BaseModel, Field, field_validator

from censor_engine.models.enums import MergeMethod
from censor_engine.models.structs.censors import Censor


class RenderingConfig(BaseModel):
    """
    This is used to handle the rendering settings of the code.

    # TODO: batch_size

    """

    batch_size: int = 4  # TODO: Multi-threading
    merge_method: MergeMethod = Field(default=MergeMethod.GROUPS)

    @field_validator("merge_method", mode="before")
    def validate_merge_method(cls, v):  # noqa: ANN001, ANN201, N805
        """Convert string input to MergeMethod enum if needed."""
        if isinstance(v, str):
            try:
                return getattr(MergeMethod, v.upper())
            except AttributeError:
                msg = f"Invalid MergeMethod value: {v}"
                raise ValueError(msg)  # noqa: B904
        return v


class AIConfig(BaseModel):
    """
    This is used for the AI model, just stuff to config it.

    # TODO: ai_model_downscale_factor

    """

    ai_model_downscale_factor: int = 1


class ReverseCensorConfig(BaseModel):
    """
    This is used for the reverse censor part of the code.

    """

    censors: list[Censor] = Field(default_factory=list)

    @field_validator("censors", mode="before")
    def validate_censors(cls, v):  # noqa: ANN001, ANN201, N805
        """Ensure censors are converted to Censor objects."""
        if not v:
            return []
        result = []
        for item in v:
            if isinstance(item, dict):
                result.append(Censor(**item))
            else:
                result.append(item)
        return result
