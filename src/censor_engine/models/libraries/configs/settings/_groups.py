from pydantic import BaseModel, Field

from censor_engine.models.libraries.configs._helper_types import Groups


class GroupSettings(BaseModel):
    persistance: Groups = Field(default_factory=list)
    merging: Groups = Field(default_factory=list)
