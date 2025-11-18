from pydantic import BaseModel, Field


class VideoConfig(BaseModel):
    """
    This config is used to handle the settings for the video pipeline.

    # TODO: censoring_fps
    # TODO: output_fps
    # TODO: persistence_groups

    """

    # Core Settings
    # NOTE: The default "-1" means to use the native FPS
    censoring_fps: int = Field(
        default=-1,
        ge=-1,
        description=(
            "NOT: IMPLEMENTED:"
            "This is the rate that the engine will apply and hold censors "
            "in length of frames. Useful for reducing processing times "
            "however the output isn't as good, also 'intense' movement "
            "can lead to parts being uncensored."
        ),
        examples=[-1, 3, 5, 10, 15],
    )
    output_fps: int = Field(
        default=-1,
        ge=-1,
        description=(
            "NOT IMPLEMENTED:"
            "This is the rate that the engine will output the video fps at. "
            "Useful for reducing processing times."
        ),
        examples=[-1, 3, 5, 10, 15],
    )

    # Video Cleaning Settings
    # # Frame Stability Config
    frame_difference_threshold: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "How much difference (percent but as a decimal) the difference in "
            "mask area there has to be for the censor to update. This is used "
            "to avoid issues where the AI model slightly changes causing "
            "The image to jitter."
        ),
        examples=[0.0, 0.05, 0.10],
    )

    # Frame Part Persistence Config
    part_frame_hold_seconds: float = Field(
        default=-1.0,
        ge=-1.0,
        description=(
            "How long parts hold their censor after they stop being detected. "
            "Used to cover cases where the AI model fails to detect the part "
            "on every other frame. The part will update positions if it exists"
            " but linger until the hold is reached."
        ),
        examples=[0.0, 0.5, 1],
    )
    persistence_groups: list[list[str]] = Field(
        default_factory=list,
        description=(
            "NOT IMPLEMENTED:"
            "I completely forgot what this does, I think it's to merge "
            "lifespans of parts."
        ),
        # examples=[-1, 3, 5, 10, 15],  # noqa: ERA001
    )
