from dataclasses import dataclass, field
import itertools
from typing import Any, Iterable, Optional, TYPE_CHECKING

import cv2
import numpy as np

from censorengine.backend.models.enums import PartState
from censorengine.lib_models.shapes import Shape
from censorengine.libs.shape_library.catalogue import shape_catalogue
from censorengine.lib_models.detectors import DetectedPartSchema


if TYPE_CHECKING:
    from censorengine.backend.constants.typing import Mask, Config


@dataclass
class Censor:
    function: str
    args: Optional[dict[str, Any]]


@dataclass
class Part:
    # From NudeNet
    part_name: str = field(init=False)
    score: float = field(init=False)
    relative_box: tuple[int, ...] = field(init=False)  # x, y, width, height

    # Derived
    box: list[tuple[int, int]] = field(init=False)  # top left, bottom right

    part_id: Iterable[int] = itertools.count(start=1)
    state: PartState = PartState.UNPROTECTED

    # Merge Groups Stuff
    merge_group_id: Optional[int] = None
    merge_group: Optional[list[str]] = None

    # Config Settings
    config: "Config" = field(init=False)
    shape: Shape = field(init=False)
    shape_name: str = field(init=False)

    censors: list[Censor] = field(default_factory=list)

    protected_shape: Optional[Shape] = None

    use_global_area: bool = True

    # Masks
    mask: "Mask" = field(init=False)
    original_mask: "Mask" = field(init=False)
    base_masks: list["Mask"] = field(default_factory=list["Mask"])

    # Information
    is_merged: bool = field(default=False)
    file_path: str = "INVALID"

    def __init__(
        self,
        detected_information: DetectedPartSchema,
        empty_mask: "Mask",
        config: "Config",
        file_path: str,
    ):
        self.config = config
        self.file_path = file_path

        # Basic
        self.part_name = detected_information.label
        self.score = detected_information.score
        self.relative_box = tuple(detected_information.relative_box)

        # Derived
        # # Box
        corrected_box = self._correct_relative_box_size()
        self._build_box(corrected_box)

        # # Part IDs
        self.part_id = next(Part.part_id)  # type: ignore

        # # Merge Groups
        self._determine_merge_groups()

        # # Config Settings
        self._determine_state()
        self._determine_shape()
        self._determine_censors()
        self._determine_meta()

        # Generate Masks
        self.mask = empty_mask
        self.original_mask = empty_mask
        self.base_masks = []
        self._get_base_mask(empty_mask)

    def _correct_relative_box_size(self):
        # Get Config Margin Data
        if not self.config.part_settings[self.part_name]:
            return self.relative_box
        config_part = self.config.part_settings[self.part_name]

        margin_data = config_part.margin

        if isinstance(margin_data, float) or isinstance(margin_data, int):
            if isinstance(margin_data, int):
                margin_data = float(margin_data)
            w_margin = margin_data
            h_margin = margin_data

        elif isinstance(margin_data, dict):
            w_margin = margin_data.get("width", 0.0)
            h_margin = margin_data.get("height", 0.0)

        else:
            w_margin = 0.0
            h_margin = 0.0

        # Get Original Data
        top_left_x, top_left_y, width, height = self.relative_box

        # Calculate New Values
        margin_width = w_margin * width
        margin_height = h_margin * height

        top_left_x -= margin_width / 2
        top_left_y -= margin_height / 2
        width += margin_width
        height += margin_height

        return [
            top_left_x,
            top_left_y,
            width,
            height,
        ]

    def __str__(self):
        return_name = f"{self.part_name}_{self.part_id}"

        if self.is_merged:
            return return_name + "_merged"
        else:
            return return_name

    def _build_box(self, box):
        top_left_x, top_left_y, width, height = box

        bottom_right_x = top_left_x + width
        bottom_right_y = top_left_y + height

        top_left = (int(top_left_x), int(top_left_y))
        bottom_right = (int(bottom_right_x), int(bottom_right_y))

        self.box = [
            top_left,  # X, Y
            bottom_right,
        ]

    def _determine_protected_shape(self):
        config_part = self.config.part_settings[self.part_name]

        # If True, Derive Shape from Settings

        protected_part_shape = config_part.shape
        if not protected_part_shape:
            protected_part_shape = "ellipse"

        self.protected_shape = protected_part_shape

    def _determine_merge_groups(self):
        if not self.config.merge_enabled:
            return

        merge_groups = self.config.merge_groups
        if not merge_groups:
            return

        for index, group in enumerate(merge_groups):
            if self.part_name in group:
                self.merge_group = group
                self.merge_group_id = index

    def _determine_state(self):
        if config_part := self.config.part_settings[self.part_name]:
            if config_part.state:
                self.state = config_part.state
            else:
                self.state = PartState.UNPROTECTED

    def _determine_shape(self):
        # Find Shape
        config_part = self.config.part_settings[self.part_name]
        shape = config_part.shape

        # Get Shape Object
        self.shape_name = shape
        self.shape = Part.get_shape_class(shape)

    def _determine_censors(self):
        self.censors = self.config.part_settings[self.part_name].censors

    def _determine_meta(self):
        self.use_global_area = self.config.part_settings[self.part_name].use_global_area

    def _get_base_mask(self, empty_mask):
        # Get Shape
        base_shape = Part.get_shape_class(self.shape.base_shape)

        # Draw Shape
        self.base_masks.append(base_shape.generate(self, empty_mask))

    def compile_base_masks(self):
        for mask in self.base_masks:
            self.add(mask)

        self.is_merged = True

    def add(self, mask):
        self.mask = cv2.add(self.mask, mask)

    def subtract(self, mask):
        self.mask = cv2.subtract(self.mask, mask)

    @staticmethod
    def normalise_mask(mask):
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # # Convert to Type
        if mask.dtype != "unit8":
            mask = mask.astype(np.uint8)

        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        return mask

    @staticmethod
    def get_shape_class(shape):
        if shape not in shape_catalogue.keys():
            return None

        return shape_catalogue[shape]()
