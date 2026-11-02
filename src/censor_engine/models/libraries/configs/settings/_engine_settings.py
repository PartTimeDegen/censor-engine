from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    model_validator,
)


class _ProcessingSettings(BaseModel):
    # TODO: Need to do these
    batch_size: NonNegativeInt = 1
    detection_downscale: NonNegativeFloat = 1


class _SegmentationSettings(BaseModel):
    use_sam_refinement: bool = False


class _DepthSettings(BaseModel):
    layers: NonNegativeInt = 0
    banding: bool = False


class _ExtraAISettings(BaseModel):
    body_segmentation: bool | _SegmentationSettings = False

    # Experimental
    clothes_segmentation: bool | _SegmentationSettings = False
    skin_segmentation: bool | _SegmentationSettings = False

    # TODO: Need to Implement
    depth: _DepthSettings | NonNegativeInt | None = None
    focused_roi: bool = False

    @model_validator(mode="after")
    def convert_depth_int_to_depth_settings(self):
        if isinstance(self.depth, int):
            self.depth = _DepthSettings(layers=self.depth)
        return self


class EngineSettings(BaseModel):
    processing: _ProcessingSettings = Field(
        default_factory=_ProcessingSettings
    )
    extra_ai_model_features: _ExtraAISettings = Field(
        default_factory=_ExtraAISettings
    )
