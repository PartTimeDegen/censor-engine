from pydantic import BaseModel

from censor_engine.models.lib_models.detectors import DetectedPartSchema

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


class Meta(BaseModel):
    hash_data: str


class CommonData(BaseModel): ...


class AIOutputData(BaseModel):
    model_name: str
    output_data: list[DetectedPartSchema]
