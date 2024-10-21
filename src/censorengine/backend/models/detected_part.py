from dataclasses import dataclass, field
from enum import IntEnum
import itertools
from typing import Any, Optional

import cv2
import numpy as np

from censorengine.backend.models.enums import PartLevel
from censorengine.lib_models.shapes import Shape
from censorengine.backend.constants.typing import Mask, Config
from censorengine.libs.shape_library.catalogue import shape_catalogue


@dataclass
class Part:
    # From NudeNet
    part_name: str = field(init=False)
    score: float = field(init=False)
    relative_box: tuple[int, int, int, int] = field(init=False)

    # Derived
    box: tuple[int, int] = field(init=False)

    part_id: int = itertools.count(start=1)
    part_level: IntEnum = PartLevel.UNPROTECTED

    # Merge Groups Stuff
    merge_group_id: Optional[int] = None
    merge_group: Optional[list[str]] = None

    # Config Settings
    config: Config = field(init=False)
    shape: str = field(init=False)
    is_default_shape: bool = field(init=False)

    censors: list[dict[str, str | dict[str, Any]]] = field(
        default_factory=list
    )
    is_default_censors: bool = field(init=False)

    protected_shape: Optional[Shape] = None

    # Masks
    mask: Mask = field(init=False)
    original_mask: Mask = field(init=False)
    base_masks: Mask = field(default_factory=list)

    def __init__(
        self,
        nude_net_info: dict[str, str | list[str | float]],
        empty_mask: Mask,
        config: Config,
    ):
        self.config = config

        # Basic
        self.part_name = nude_net_info["class"]
        self.score = nude_net_info["score"]
        self.relative_box = tuple(nude_net_info["box"])

        # Derived
        # # Box
        corrected_box = self._correct_relative_box_size()
        self._build_box(corrected_box)

        # # Part IDs
        self.part_id = next(Part.part_id)

        # # Part Level
        self._determine_part_level()

        # # Merge Groups
        self._determine_merge_groups()

        # # Config Settings
        self._determine_shape()
        self._determine_censors()

        # Generate Masks
        self.mask = empty_mask
        self.original_mask = empty_mask
        self.base_masks = []
        self._get_base_mask(empty_mask)

    def _correct_relative_box_size(self):
        # Get Config Margin Data
        config_part = self.config["information"].get(self.part_name)

        if config_part and config_part.get("margin", False):
            margin_data = self.config["information"][self.part_name]["margin"]

            if isinstance(margin_data, float):
                w_margin = margin_data
                h_margin = margin_data
            else:
                w_margin = margin_data.get("width", 0.0)
                h_margin = margin_data.get("height", 0.0)

        else:
            w_margin = self.config["information"]["defaults"].get(
                "margin", 0.0
            )
            h_margin = self.config["information"]["defaults"].get(
                "margin", 0.0
            )

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
        return f"{self.part_name}_{self.part_id}"

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

    def _determine_part_level(self):
        config_info = self.config["information"]

        if not config_info.get(self.part_name):
            return

        if config_info[self.part_name].get("protected"):
            self.part_level = PartLevel.PROTECTED
            self._determine_protected_shape()
        elif config_info[self.part_name].get("revealed"):
            self.part_level = PartLevel.REVEALED

    def _determine_protected_shape(self):
        config_part = self.config["information"][self.part_name]
        protected_part_shape = config_part.get("protected")

        # If True, Derive Shape from Settings
        if isinstance(protected_part_shape, bool):
            config_defaults = self.config["information"]["defaults"]

            protected_part_shape = config_part.get("shape")
            if not protected_part_shape:
                protected_part_shape = config_defaults.get("protected_shape")
            if not protected_part_shape:
                protected_part_shape = config_defaults.get("shape")
            if not protected_part_shape:
                protected_part_shape = "ellipse"

        self.protected_shape = protected_part_shape

    def _determine_merge_groups(self):
        config_info = self.config["information"]

        config_merge_info = config_info.get("merging")
        if not config_merge_info:
            return

        merge_groups = config_merge_info.get("merge_groups")
        if not merge_groups:
            return

        for index, group in enumerate(merge_groups):
            if self.part_name in group:
                self.merge_group = group
                self.merge_group_id = index

    def _determine_shape(self):
        # Get Config
        config_info = self.config["information"]

        # Find Shape
        try:
            shape = config_info[self.part_name]["shape"]
            self.is_default_shape = True
        except KeyError:
            shape = config_info["defaults"]["shape"]
            self.is_default_shape = False

        # Get Shape Object
        self.shape = Part.get_shape_class(shape)

    def _determine_censors(self):
        config_info = self.config["information"]

        try:
            censors = (config_info[self.part_name]["censors"], False)
        except KeyError:
            if self.part_level == PartLevel.REVEALED:
                self.censors = None
                self.is_default_censors = True
                return

            censors = (config_info["defaults"]["censors"], True)

        self.censors, self.is_default_censors = censors

    def _get_base_mask(self, empty_mask):
        # Get Shape
        base_shape_name = self.shape.base_shape
        base_shape = Part.get_shape_class(base_shape_name)

        # Draw Shape
        self.base_masks.append(base_shape.generate(self.box, empty_mask))

    def compile_base_masks(self):
        for mask in self.base_masks:
            self.add(mask)

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
