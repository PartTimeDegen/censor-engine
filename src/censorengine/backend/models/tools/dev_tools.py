from dataclasses import dataclass
import os

import cv2

from censorengine.backend.models.structures.detected_part import Part


@dataclass(slots=True)
class DevTools:
    output_folder: str

    def __post_init__(self):
        if not os.path.exists(".dev"):
            os.makedirs(".dev")

    def dev_decompile_masks(self, parts: list[Part], subfolder: str | None = None):
        # Main Folder

        current_path = os.path.join(".dev", self.output_folder)

        # Sub Folder
        if subfolder:
            current_path = os.path.join(current_path, subfolder)

        # Generate Folder
        if not os.path.exists(current_path) and len(parts) != 0:
            os.makedirs(current_path)

        # Include Parts
        for part in parts:
            path = os.path.join(current_path, part.get_name_and_id() + ".jpg")
            cv2.imwrite(path, part.mask)
