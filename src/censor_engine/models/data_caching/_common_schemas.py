from pydantic import BaseModel

from censor_engine.models.library_models.detectors.schemas import (
    DetectedPart,
)

"""
This holds the schemas for the caching mechanism

Cache System

    cache/
        [file_path_in_uncensored]/
            meta.json
            frames/
                1.json # First frame or just image, for ease
                2.json
                3.json
                ...

The end goal is something that can be redundant and fast

"""


class MetaData(BaseModel):
    hash_data: str


class AIOutputData(BaseModel):
    model_name: str
    output_data: list[DetectedPart]
    frame: int = 1


class CacheData(BaseModel):
    cache_data: list[AIOutputData]
