from pydantic import BaseModel, Field


class SupplementaryAIConfig(BaseModel):
    layers: int = 0
    body_segmentation: bool = True

    # TODO
    clothes_segmentation: bool = False
    focused_roi: bool = False


class AIConfig(BaseModel):
    """
    This is used for the AI model, just stuff to config it.

    # TODO: ai_model_downscale_factor

    """

    # TODO
    ai_model_downscale_factor: int = 1

    # Enabled
    detections_enabled: list[str] | str = Field(
        default="all",
        description="List of enabled parts",
        examples=[
            "all",  # TODO: Account for this, it's in the code just not here
            ["FEMALE_BREAST_EXPOSED", "FEMALE_BREAST_COVERED"],
            [
                "FEMALE_BREAST_EXPOSED",
                "FEMALE_BREAST_COVERED",
                "FEMALE_GENITALIA_EXPOSED",
                "FEMALE_GENITALIA_COVERED",
            ],
        ],
    )
    extras: SupplementaryAIConfig = Field(
        default_factory=SupplementaryAIConfig
    )
