from dataclasses import dataclass, field
import os

import cv2
from itertools import count

from censor_engine.detected_part.base import Part


@dataclass(slots=True)
class DevTools:
    output_folder: str
    main_files_path: str
    using_full_output_path: bool

    counter: int = field(default_factory=count().__next__, init=False)

    def __post_init__(self):
        if not os.path.exists(".dev"):
            os.makedirs(".dev")

    def dev_decompile_masks(
        self,
        parts: list[Part] | Part,
        subfolder: str | None = None,
    ):
        # Main Folder
        current_path = os.path.join(
            ".dev",
            self.output_folder.replace(self.main_files_path, "", 1)[1:],
        )

        # Sub Folder
        current_path = current_path.split(os.sep)
        current_path[-1] = f"{self.counter}_" + current_path[-1]
        current_path = os.path.join(*current_path)

        if subfolder:
            current_path = os.path.join(current_path, subfolder)

        # Generate Folder
        if isinstance(parts, Part):
            parts = [parts]
        if not os.path.exists(current_path) and len(parts) != 0:
            os.makedirs(current_path, exist_ok=True)

        # Include Parts
        for part in parts:
            path = os.path.join(current_path, part.get_name_and_id() + ".jpg")
            cv2.imwrite(path, part.mask)
            # print(f"=={subfolder}::{part.get_name_and_id()}")
