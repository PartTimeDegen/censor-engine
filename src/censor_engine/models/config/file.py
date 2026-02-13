from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class FileConfig(BaseModel):
    """
    This config handles the file paths.

    """

    file_prefix: str = ""
    file_suffix: str = ""

    uncensored_folder: Path = Field(default=Path("uncensored"))
    censored_folder: Path = Field(default=Path("censored"))

    # Optional validator for ensuring conversion from str to Path
    @field_validator("uncensored_folder", "censored_folder", mode="before")
    def ensure_path(cls, v):  # noqa: ANN001, N805
        """
        This ensures the paths are Path.

        """
        if isinstance(v, str):
            return Path(v)
        return v
