from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import itertools
import statistics
from typing import Optional

import cv2
import numpy as np
from typing import TYPE_CHECKING

from censorengine.backend.models.debugger import (
    Debugger,
    DebugLevels,
)
from censorengine.backend.models.enums import PartState, ShapeType

if TYPE_CHECKING:
    from censorengine.backend.constants.typing import (
        CVImage,
        Config,
    )
from censorengine.backend.models.detected_part import Part
from censorengine.libs.style_library.catalogue import style_catalogue

from censorengine.libs.detector_library.catalogue import (
    enabled_detectors,
    enabled_determiners,
)
from censorengine.lib_models.detectors import DetectedPartSchema


@dataclass
class CensorManager:
    # Parts
    detected_parts: list[Optional[dict[str, str]]] = field(default_factory=list)
    parts: list[Optional[Part]] = field(default_factory=list)

    # Common Masks
    empty_mask: "CVImage" = field(init=False)

    # File Info
    file_original_image: "CVImage" = field(init=False)
    file_image: "CVImage" = field(init=False)
    file_loc: str = field(init=False)
    file_image_name: str = field(init=False)
    force_png: bool = False

    # Manager Info
    config: "Config" = field(init=False)

    # Image Determiners
    extracted_information: dict[str, str] = field(default_factory=dict)

    # Debugger
    debugger: Debugger = field(init=False)

    # Stats
    # TODO: Expand on part_type_id, make an Enum and can include part "area"
    # (ie group exposed and covered variants)
    # part_type_id: int
    # part_type_counts: int

    def __init__(
        self,
        file_path: str,
        config: "Config",
        index_text: str = "",
    ):
        self.config = config
        Part.part_id = itertools.count(start=1)
        self.detected_parts = []
        self.extracted_information = {}

        # File Stuff
        self.file_loc = file_path
        self.file_image_name = file_path.split("/")[-1]

        # NOTE: This may produce
        #       "libpng warning: iCCP: known incorrect sRGB profile" errors
        #       I tried suppressing them but it doesn't work
        self.file_original_image = cv2.imread(file_path)
        self.file_image = cv2.imread(file_path)

        # Declare Start
        if index_text != "":
            index_text += " "
        print()
        print(f'{index_text}Censoring: "{self.file_image_name}"')

        # Debug
        self.debugger = Debugger("Censor Manager", level=DebugLevels.DETAILED)
        self.debugger.time_total_start()
        self.debugger.display_onnx_info()

        # Empty Mask
        self.debugger.time_start("Create Empty Mask")
        self.empty_mask = self._create_empty_mask()
        self.debugger.time_stop()

        # Determine Image
        self.debugger.time_start("Determine Image")
        self._determine_image()
        self.debugger.time_stop()

        # Detect Parts for Image
        self.debugger.time_start("Detect Parts")
        self._detect_image()
        self.debugger.time_stop()

    # Private
    def _create_empty_mask(self, inverse: bool = False):
        """
        This function acts as a factory for empty masks, due to 1) bad copying
        issues, and 2) because it's not a one-liner

        :param bool inverse: Inverses the mask to make it white on black, not black on white, defaults to False

        # TODO: Cache this if it's made

        """
        if inverse:
            return Part.normalise_mask(
                np.ones(
                    self.file_image.shape,
                    dtype=np.uint8,
                )
                * 255
            )

        return Part.normalise_mask(
            np.zeros(
                self.file_image.shape,
                dtype=np.uint8,
            )
        )

    def _determine_image(self):
        """
        This method is used for the enabled determiners, ie ai_models that
        extract information from the photo (that don't provide locations).

        Oirignally meant for `nsfw_detector` however my computer can't run
        tensorflow but I will keep it here.

        """
        # Collect Detected Data
        self.extracted_information = {
            determiner.model_name: determiner.determine_image(self.file_loc)
            for determiner in enabled_determiners
        }

    def _detect_image(self):
        """
        #TODO : Write me

        """
        # Collect Detected Data
        with ThreadPoolExecutor() as executor:
            detected_parts = list(
                executor.map(
                    lambda detector: detector.detect_image(self.file_loc),
                    enabled_detectors,
                )
            )

        self.detected_parts = list(itertools.chain(*detected_parts))

    def _create_parts(self):
        """
        This function creates the list of Parts for CensorEngine to keep track
        of. # TODO: Update me to account for split with _detection

        Method:
            1)  It will create an empty list and also find the enabled parts
            2)  It will then collect a list of all of the parts from all of the
                enable AI models/detectors.
            3)  It will then use the parts list to make `Part` objects from the
                found parts, while also discarding any that aren't enabled.
            4)  It then will filter for `None` values (by product of the
                function)

        Notes:
            -   The reason a Map/Filter function is used is because this part
                of the code takes a while to run, using map/filter reduces the
                time massively, as ugly as it looks (I did try to learn it up
                but there's only so much makeup you can put on a pig).

            -   NudeNet (and I assume others) were found to be 98% of the time
                taken for this to run, so it's slow but it's because of the
                package/model itself, not the rest of the code. It's pretty
                much optimised as much as it can be (even the Part creation in
                total was only 0.005s, which is nothing)

        """
        # Acquire Settings
        self.parts = []
        config_parts_enabled = self.config.parts_enabled

        # Map and Filter Parts for Missing Information
        def add_parts(detect_part: DetectedPartSchema):
            if detect_part.label not in config_parts_enabled:
                return

            return Part(
                detected_information=detect_part,
                empty_mask=self._create_empty_mask(),
                config=self.config,
                file_path=self.file_loc,
            )

        self.parts = list((map(add_parts, self.detected_parts)))
        self.parts = list(filter(lambda x: x is not None, self.parts))

    def _merge_parts_if_in_merge_groups(self):
        if not self.config.merge_enabled:
            return

        merge_groups = self.config.merge_groups
        if not merge_groups:
            return

        full_parts = self.parts
        for index, part in enumerate(full_parts):
            if not part.merge_group:
                continue

            for other_part in full_parts[index + 1 :]:
                if part == other_part:
                    continue
                if other_part.part_name not in part.merge_group:
                    continue

                part.base_masks.append(other_part.base_masks[0])
                self.parts.remove(other_part)

            part.compile_base_masks()

    def _apply_mask_shapes(self):
        for part in self.parts:
            # For Single Shapes
            if not part.is_merged:
                shape_single = Part.get_shape_class(part.shape.single_shape)

                if not shape_single:
                    raise KeyError("Missing single shape")

                part.mask = shape_single.generate(part, self.empty_mask.copy())
                continue

            # For Advanced Shaoes
            match part.shape.shape_type:
                case ShapeType.BASIC:
                    pass

                case ShapeType.JOINT:
                    part.mask = part.shape.generate(part, self.empty_mask.copy())

                case ShapeType.BAR:
                    shape_joint = Part.get_shape_class("joint_ellipse")
                    part.mask = shape_joint.generate(part, self.empty_mask.copy())
                    part.mask = part.shape.generate(part, self.empty_mask.copy())

    def _process_overlaps_for_masks(self):
        full_parts = sorted(self.parts, key=lambda x: (-x.state, x.part_name))[::-1]

        for index, target_part in enumerate(full_parts):
            if target_part not in self.parts:
                continue

            for comp_part in full_parts[index + 1 :]:
                if target_part not in self.parts or comp_part not in self.parts:
                    continue
                # NOTE: This is done in respect to Target

                # Quality of Life Booleans
                compared_settings = {
                    "censors": target_part.censors == comp_part.censors,
                    "states": target_part.state == comp_part.state,
                }

                is_comp_higher = target_part.state <= comp_part.state

                # Flow Chart
                # # MATCHING
                if all(compared_settings.values()):
                    # ALL MATCHING CRITERIA
                    # Combine Parts
                    target_part.add(comp_part.mask)
                    self.parts.remove(comp_part)

                # # PROTECTED
                elif (
                    target_part.state == PartState.PROTECTED
                    or comp_part.state == PartState.PROTECTED
                ):
                    if compared_settings["states"]:
                        comp_part.subtract(target_part.mask)
                    elif target_part.state == PartState.PROTECTED:
                        comp_part.subtract(target_part.mask)
                    elif comp_part.state == PartState.PROTECTED:
                        target_part.subtract(comp_part.mask)

                # # REVEALED
                elif target_part.state == PartState.REVEALED:
                    if is_comp_higher:
                        comp_part.subtract(target_part.mask)
                    else:
                        target_part.subtract(comp_part.mask)

                # # UNPROTECTED
                elif target_part.state == PartState.UNPROTECTED:
                    if is_comp_higher and compared_settings["censors"]:
                        comp_part.add(target_part.mask)
                        self.parts.remove(target_part)

                    elif is_comp_higher:
                        target_part.subtract(comp_part.mask)

    def _handle_reverse_censor(self):
        if not self.config.reverse_censor_enabled:
            return

        # Create Mask
        base_mask_reverse = self._create_empty_mask(inverse=True)
        for part in self.parts:
            base_mask_reverse = cv2.subtract(base_mask_reverse, part.mask)

        # Apply Censors
        contour = cv2.findContours(
            image=base_mask_reverse,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        for censor in self.config.reverse_censors[::-1]:
            censor_object = style_catalogue[censor.function]()
            censor_object.change_linetype(enable_aa=False)
            censor_object.using_reverse_censor = True

            arguments = [
                self.file_image,
                contour,
            ]
            if censor_object.style_type == "dev":
                arguments.append(part)

            self.file_image = censor_object.apply_style(
                *arguments,
                **censor.args,
            )

    def _apply_censors(self):
        parts = sorted(self.parts, key=lambda x: (x.state, x.part_name))

        for part in parts:
            if not part.censors or not part:
                continue

            # if part.use_global_area:
            part_contour = cv2.findContours(
                image=part.mask,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )  # TODO: reduce image size to just part

            # Save Copy for Effects
            processed_image = self.file_image.copy()

            # Gather default args
            arguments = [
                processed_image,
                part_contour,
            ]

            # Reversed to represent YAML order
            for censor in part.censors[::-1]:
                # Acquire Function
                censor_object = style_catalogue[censor.function]()

                # Turn on AA
                censor_object.change_linetype(enable_aa=True)

                # Handle Args
                loop_args = arguments  # Optimisation
                if censor_object.style_type == "dev":
                    arguments.append(part)

                # Apply Censor
                processed_image = censor_object.apply_style(
                    *loop_args,
                    **censor.args,
                )

            # Apply Potential Feather Fade
            if not part.fade_percent:
                continue

            # # Get Mask
            contour_mask = cv2.drawContours(
                np.zeros(self.file_image.shape, dtype=np.uint8),
                part_contour[0],
                -1,
                (255, 255, 255),
                -1,
            )

            # # Convert Mask to Right Format
            contour_mask = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY)
            contour_mask = contour_mask.astype(float) / 255.0

            # # Calculate Feather Effect based on Contour
            _, _, w, h = cv2.boundingRect(part_contour[0][0])
            fade_percent = np.clip(int(part.fade_percent), 0, 100)
            max_dim = max(w, h)
            feathering_amount = int((fade_percent / 100.0) * max_dim)
            kernel_size = min(51, max(3, feathering_amount // 2 * 2 + 1))

            # # Apply Gaussian blur for feathering
            contour_mask = cv2.erode(
                contour_mask,
                np.ones((kernel_size, kernel_size), np.uint8),
                iterations=1,
            ).astype(float)
            feathered_mask = cv2.GaussianBlur(
                contour_mask, (kernel_size, kernel_size), 0
            )

            # # Convert the mask back to 3-channel for blending
            feathered_mask = cv2.merge([feathered_mask] * 3)

            # # Blend the images using the feathered mask
            feathered_mask *= 1.0
            merged_image = (
                feathered_mask * processed_image
                + (1 - feathered_mask) * self.file_image
            )

            self.file_image = merged_image

    # Lists of Parts
    def get_list_of_parts_total(self, search: Optional[dict[str, str]] = None):
        if not search:
            return self.parts

        list_matching_parts_attributes = [
            part
            for part in self.parts
            if all(
                part.__dict__.get(key) and (str(part.__dict__.get(key)) == str(value))
                for key, value in search.items()
            )
        ]

        return list_matching_parts_attributes

    # Debugger
    def flush_debugger(self):
        return self.debugger

    # Static
    @staticmethod
    def get_statistics(
        durations: list[float],
    ) -> dict[str, float]:
        collection = {}
        collection["mean"] = statistics.mean(durations)
        collection["median"] = statistics.median(durations)
        collection["max"] = max(durations)
        collection["min"] = min(durations)
        collection["range"] = max(durations) - min(durations)

        if len(durations) != 1:
            collection["stdev"] = statistics.stdev(durations)
            if collection["stdev"] and collection["mean"] != 0.0:
                collection["coefficient_of_variation"] = (
                    collection["stdev"] / collection["mean"]
                )

        return collection

    def return_output(self):
        return self.file_image

    # Public
    def start(self):
        # Create Parts
        self.debugger.time_start("Create Parts")
        self._create_parts()
        self.debugger.time_stop()

        # Merge Parts
        self.debugger.time_start("Merge Parts")
        self._merge_parts_if_in_merge_groups()
        self.debugger.time_stop()

        # Handle More Advanced Parts (i.e., Bars and Joints)
        self.debugger.time_start("Handle Advanced Shapes")
        self._apply_mask_shapes()
        self.debugger.time_stop()

        # Test Parts for Overlap
        self.debugger.time_start("Process Overlaps")
        self._process_overlaps_for_masks()
        self.debugger.time_stop()

        # Generate and Apply Reverse Censor
        self.debugger.time_start("Generate Reverse Censor")
        self._handle_reverse_censor()
        self.debugger.time_stop()

        # Apply Censors
        self.debugger.time_start("Apply Censors")
        self._apply_censors()
        self.debugger.time_stop()

        # DEBUG: Times
        self.debugger.time_total_end()
        self.debugger.display_times()
