import copy
from dataclasses import dataclass, field

import cv2

from censor_engine._typing import MaskImage
from censor_engine.api.masks import MaskContext
from censor_engine.libraries.registries import MaskRegistry
from censor_engine.models.models_library.masks import Mask


@dataclass(slots=True)
class MaskManager:
    """
    Manages mask generation, storage, and composition for a Part.

    This class handles the lifecycle of masks associated with a part,
    including:

    * Creating the base mask.
    * Creating an optional protection mask.
    * Storing the original generated mask.
    * Maintaining a mutable working mask.
    * Managing additional mask layers.
    * Combining and modifying masks using OpenCV operations.

    Attributes:
        mask_name (str):
            Name of the primary mask registered in the mask registry.

        protection_mask_name (str | None):
            Name of the protection mask registry entry.
            If `None`, a copy of the primary mask is used.

    """

    mask_name: str
    protection_mask_name: str | None
    image_shape: tuple[int, int]

    # Generated
    _obj_mask: Mask = field(init=False)
    _obj_mask_protected: Mask = field(init=False)

    # Mask Arrays
    _original_mask: MaskImage = field(init=False)
    _current_mask: MaskImage = field(init=False)
    _layers_of_mask: list[MaskImage] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """
        Initialize mask objects and generate the initial masks.

        Loads the configured mask classes from the registry, creates the
        original mask, initializes the layer stack, and creates the
        mutable current mask.

        Returns:
            None

        """
        # Base Mask
        self._obj_mask = self.get_mask_class(self.mask_name)

        # Protection Mask if Used
        if self.protection_mask_name is not None:
            self._obj_mask_protected = self.get_mask_class(
                self.protection_mask_name
            )
        else:
            self._obj_mask_protected = copy.copy(self._obj_mask)

        # Generate Masks
        mask_context = MaskContext(
            part=self,
            empty_mask=Mask.create_empty_mask(self.image_shape),
        )
        self._original_mask = self._obj_mask.generate(mask_context)
        self._layers_of_mask = [self._original_mask]

        self._current_mask = self._original_mask.copy()

    @staticmethod
    def get_mask_class(mask: str) -> Mask:
        """
        Retrieve and instantiate a mask class from the registry.

        Args:
            mask (str):
                Name of the mask registered in the mask registry.

        Returns:
            Mask:
                A newly instantiated mask object.

        Raises:
            ValueError:
                If the requested mask name is not registered.

        """
        masks = MaskRegistry.get_all()
        if mask not in masks:
            msg = f"Mask {mask} does not Exist! {[masks.keys()]}"
            raise ValueError(msg)

        return masks[mask]()

    def add_to_current_mask(self, mask: MaskImage) -> None:
        """
        Add a mask to the current mask.

        Performs a saturated pixel-wise addition using OpenCV.

        Args:
            mask (MaskImage):
                Mask image to add to the current mask.

        Returns:
            None

        """
        self._current_mask = cv2.add(self._current_mask, mask)  # type: ignore

    def subtract_from_current_mask(self, mask: MaskImage) -> None:
        """
        Subtract a mask from the current mask.

        Performs a saturated pixel-wise subtraction using OpenCV.

        Args:
            mask (MaskImage):
                Mask image to subtract from the current mask.

        Returns:
            None

        """
        self._current_mask = cv2.subtract(self._current_mask, mask)  # type: ignore

    def compile_base_masks(self) -> None:
        """
        Compile all stored mask layers into the current mask.

        Iterates through all masks in `_layers_of_mask` and adds them
        to the current mask. This can be used to rebuild the working mask
        from its constituent layers.

        Returns:
            None

        """
        for mask in self._layers_of_mask:
            self.add_to_current_mask(mask)
