from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

import cv2
import numpy as np

from censor_engine.libs.registries import ShapeRegistry
from censor_engine.models.config import PartSettingsConfig
from censor_engine.models.lib_models.shapes import Shape
from censor_engine.models.config import Config
from censor_engine.models.structs.part_areas import PartArea


if TYPE_CHECKING:
    from censor_engine.typing import Mask


@dataclass(slots=True)
class Part:
    # Input
    part_name: str
    part_id: int
    score: float
    relative_box: tuple[int, int, int, int]  # x, y, width, height
    config: Config

    file_uuid: UUID
    image_shape: tuple[int, int, ...]  # type: ignore

    # Internal
    # # Found
    part_settings: PartSettingsConfig = field(init=False)

    # # Meta
    is_merged: bool = False
    merge_group_id: int | None = None
    persistence_group_id: int | None = None

    # # Generated
    part_area: PartArea = field(init=False)
    merge_group: list[str] = field(default_factory=list, init=False)
    persistence_group: list[str] = field(default_factory=list, init=False)

    shape_name: str = field(default="NOT_SET", init=False)
    shape_object: Shape = field(default_factory=Shape, init=False)
    protected_shape_object: Shape = field(default_factory=Shape, init=False)

    # # Masks
    mask: "Mask" = field(init=False)
    original_mask: "Mask" = field(init=False)
    base_masks: list["Mask"] = field(default_factory=list, init=False)

    def __post_init__(self):
        # Connect Settings
        self.part_settings = self.config.censor_settings.parts_settings[self.part_name]

        # Derived
        # # Box
        self._correct_relative_box_size()

        # # Merge Groups
        for index, group in enumerate(
            self.config.censor_settings.merge_settings.merge_groups, start=1
        ):
            if self.part_name in group:
                self.merge_group = group
                self.merge_group_id = index

        for index, group in enumerate(
            self.config.video_settings.persistence_groups, start=1
        ):
            if self.part_name in group:
                self.persistence_group = group
                self.persistence_group_id = index

        # Determine Shape
        self.shape_object = Part.get_shape_class(self.part_settings.shape)
        self.shape_name = self.shape_object.shape_name

        # Determine Protected Shape
        if protected_part_shape := self.part_settings.protected_shape:
            protected_shape = Part.get_shape_class(protected_part_shape)
        else:
            protected_part_shape = self.part_settings.shape
            protected_shape = Part.get_shape_class(protected_part_shape)
        self.protected_shape_object = protected_shape

        # Generate Masks
        self.original_mask = Part.create_empty_mask(self.image_shape)

        base_shape = Part.get_shape_class(self.shape_object.base_shape)
        self.mask = base_shape.generate(self, Part.create_empty_mask(self.image_shape))
        self.base_masks = [self.mask]

    def __str__(self) -> str:
        return f"{self.part_name}_{self.part_id}{"_merged" if self.is_merged else ""}"

    def __repr__(self) -> str:
        return self.part_name

    def _correct_relative_box_size(self) -> None:
        margin_data = self.part_settings.margin

        if isinstance(margin_data, (float, int)):
            margin_data = float(margin_data)
            w_margin = margin_data
            h_margin = margin_data

        elif isinstance(margin_data, dict):
            w_margin = margin_data.get("width", 0.0)
            h_margin = margin_data.get("height", 0.0)

        else:
            w_margin = h_margin = 0.0

        # Get Original Data
        top_left_x, top_left_y, width, height = self.relative_box

        # Calculate New Values
        margin_width, margin_height = w_margin * width, h_margin * height

        top_left_x -= margin_width / 2
        top_left_y -= margin_height / 2
        width += margin_width
        height += margin_height

        # Format Box Values to Int and Make to Standard Box Format
        new_relative_box = (int(top_left_x), int(top_left_y), int(width), int(height))

        self.part_area = PartArea(
            new_relative_box,
            self.part_settings.video_part_search_region,
            self.image_shape[:2],
        )

    # Public Methods
    # # Naming Methods
    def get_id_name_and_merged(self) -> str:
        return f"{self.part_id}_{self.part_name}{"_merged" if self.is_merged else ""}"

    def get_name_and_merged(self) -> str:
        return f"{self.part_name}{"_merged" if self.is_merged else ""}"

    def get_name(self) -> str:
        return self.part_name

    def get_name_and_id(self) -> str:
        return f"{self.part_name}_{self.part_id}"

    # # Mask Equations
    def compile_base_masks(self) -> None:
        for mask in self.base_masks:
            self.add(mask)

        self.is_merged = True

    def add(self, mask: "Mask") -> None:
        self.mask = cv2.add(self.mask, mask)

    def subtract(self, mask: "Mask") -> None:
        self.mask = cv2.subtract(self.mask, mask)

    @staticmethod
    def normalise_mask(mask: "Mask") -> "Mask":
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # # Convert to Type
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        return mask

    @staticmethod
    def get_shape_class(shape: str) -> Shape:
        shapes = ShapeRegistry.get_all()
        if shape not in shapes:
            raise ValueError(f"Shape {shape} does not Exist! {[shapes.keys()]}")

        return shapes[shape]()

    @staticmethod
    def create_empty_mask(
        image_shape: tuple[int, int, int],
        inverse: bool = False,
    ):
        """
        This function acts as a factory for empty masks, due to 1) bad copying
        issues, and 2) because it's not a one-liner

        :param bool inverse: Inverses the mask to make it white on black, not black on white, defaults to False

        # TODO: Cache this if it's made

        """
        if inverse:
            return Part.normalise_mask(
                np.ones(image_shape, dtype=np.uint8) * 255  # type: ignore
            )
        return Part.normalise_mask(np.zeros(image_shape, dtype=np.uint8))
