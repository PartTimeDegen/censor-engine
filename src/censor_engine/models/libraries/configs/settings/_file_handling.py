from pathlib import Path

from pydantic import BaseModel, Field


class _Files(BaseModel):
    prefix: str = ""
    suffix: str = ""


class _Folders(BaseModel):
    uncensored: Path = Path("uncensored")
    censored: Path = Path("censored")


class FileHandingSettings(BaseModel):
    files: _Files = Field(default_factory=_Files)
    folders: _Folders = Field(default_factory=_Folders)
