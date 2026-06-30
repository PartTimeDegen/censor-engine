from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt


class _ProcessingSettings(BaseModel):
    batch_size: NonNegativeInt = 1
    detection_downscale: NonNegativeFloat = 1


class _ExtraAISettings(BaseModel):
    censor_layers: NonNegativeInt = 0
    body_segmentation: bool = True

    # TODO: Need to Implement
    clothes_segmentation: bool = False
    focused_roi: bool = False


class EngineSettings(BaseModel):
    processing: _ProcessingSettings = Field(
        default_factory=_ProcessingSettings
    )
    extra_ai_model_features: _ExtraAISettings = Field(
        default_factory=_ExtraAISettings
    )
