import os
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

import cv2

from censor_engine.detected_part import Part


@dataclass(slots=True)
class DevTools:
    output_folder: Path
    main_files_path: Path
    using_full_output_path: bool

    counter: int = field(default_factory=count().__next__, init=False)

    def __post_init__(self):
        if not os.path.exists(".dev"):
            os.makedirs(".dev")

    def dev_decompile_masks(
        self,
        parts: list[Part] | Part,
        subfolder: str | None = None,
    ) -> None:
        # Main folder path, relative to main_files_path, prefixed with ".dev"
        current_path = Path(".dev") / self.output_folder.relative_to(
            self.main_files_path,
        )

        # Modify last part of path by prepending counter + "_"
        parts_list = list(current_path.parts)
        parts_list[-1] = f"{self.counter}_" + parts_list[-1]
        current_path = Path(*parts_list)

        # Add subfolder if provided
        if subfolder:
            current_path = current_path / subfolder

        # Ensure parts is a list
        if isinstance(parts, Part):
            parts = [parts]

        # Create directory if it doesn't exist and there are parts
        if not current_path.exists() and parts:
            current_path.mkdir(parents=True, exist_ok=True)

        # Save mask images for each part
        for part in parts:
            file_path = current_path / f"{part.get_name_and_id()}.jpg"
            cv2.imwrite(str(file_path), part.mask)
            # print(f"=={subfolder}::{part.get_name_and_id()}")
