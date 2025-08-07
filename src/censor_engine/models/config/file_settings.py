from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FileConfig:
    file_prefix: str = ""
    file_suffix: str = ""

    uncensored_folder: Path = Path("uncensored")
    censored_folder: Path = Path("censored")

    def __post_init__(self):
        # Type Narrow to Float
        if isinstance(self.uncensored_folder, str):
            self.uncensored_folder = Path(self.uncensored_folder)
        if isinstance(self.censored_folder, str):
            self.censored_folder = Path(self.censored_folder)
