from functools import reduce

import cv2
import numpy as np

from censor_engine.models.core.tools.mask_helpers import (
    edge_smoothing,
    hole_filling,
    island_cleanup,
)
from censor_engine.models.libraries.ai_models._enums import OutputType
from censor_engine.models.libraries.ai_models.output_schemas import (
    DetectorOutput,
)
from censor_engine.typing import MaskImage


class AdvancedMaskStrategy:
    @staticmethod
    def skin_mask(  # noqa: PLR0917
        person_mask: MaskImage,
        outfit_mask: MaskImage,
        islands_min_area: int = 200,
        islands_relative_threshold: float = 0.1,
        kernal_size_pre: int = 7,
        kernal_size_string_removal: int = 7,
    ) -> MaskImage:
        """
        This Mask is the skin mask, it selects only the skin of a person.

        This mask is formed from using a body segmentation mask then
        subtracting a clothes segmentation mask, which then provides only the
        clothes.

        Due to differences in the masks, extra cleaning is required.

        Notes:
            -   The Mask has an undercut issue, seems to be due to the
                conservative nature of the outfit mask.
            -   Sheer materials seem to get labelled at clothes because they
                technically are, extra work needs to be done to fix this.
            -   The mask of the clothes seems to "leak" way too much, it might
                be better to sample it then use something like SAM2 to properly
                get the clothing.

        Args:
            person_mask: Mask of the person, from YoloSeg
            outfit_mask: Mask of the outfit, from HumanParsing
            islands_min_area: Minimum size of islands, not really used because
                of varying image size. Defaults to 200.
            islands_relative_threshold: Relative size of smallest islands from
                biggest island. Defaults to 0.1.
            kernal_size_pre: Kernal size of the trimming from body mask.
                Defaults to 7.
            kernal_size_string_removal: Kernal size of the mask string
                cleaning. Defaults to 7.

        Returns:
            Skin Mask

        """
        mask = cv2.subtract(person_mask, outfit_mask)

        # Be Conservative in Processing
        kernal_size_pre = 7
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernal_size_pre, kernal_size_pre),
        )
        mask = cv2.erode(mask, kernel, iterations=1)

        # Fill Holes due to HumanParse Messing Up
        mask = hole_filling.fill_holes_within_mask(mask)  # type: ignore

        # Remove Stringy Bits (e.g., skin bigger than clothes)
        # NOTE: Because the masks aren't from the same model, the clothing mask
        #       can be larger than the body mask, which creates an outline
        #       effect, this removes that.
        mask = edge_smoothing.remove_stringy_parts_of_mask(
            mask,
            kernal_size=kernal_size_string_removal,
        )

        # Only Get Largest Part
        mask = island_cleanup.filter_to_largest_connected_islands(
            mask,
            min_area=islands_min_area,
            relative_threshold=islands_relative_threshold,
        )  # type: ignore

        # Handle Edges
        mask = edge_smoothing.smooth_edges(
            mask,
            kernal_size=kernal_size_string_removal,
        )

        # Fix Undercut
        kernel_undercut = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernal_size_pre, kernal_size_pre),
        )
        mask = cv2.dilate(mask, kernel_undercut, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_undercut)

        # Use the Original Body Mask to Get Rid of the Excess Mask
        return cv2.bitwise_and(person_mask, mask)  # type: ignore


class InferredDetectionParts:
    prefix: str = "INFERRED"

    def _find_detections_from_label(
        self,
        detection_outputs: list[DetectorOutput],
        labels: tuple[str, ...] | str,
    ) -> list[DetectorOutput]:
        if isinstance(labels, str):
            labels = (labels,)

        return [
            detection
            for detection in detection_outputs
            if detection.label in labels
        ]

    def _flatten_masks_of_detections(
        self, detections: list[DetectorOutput]
    ) -> DetectorOutput:
        # TODO: Generalise this to accept any
        if not detections:
            msg = "Missing Detections"
            raise ValueError(msg)

        # TODO: There must be a better way to do this copying
        return DetectorOutput(
            origin=detections[0].origin,
            label=detections[0].label,
            output_type=detections[0].output_type,
            score=np.mean(
                [
                    detection.score
                    for detection in detections
                    if detection.score
                ]
            ),
            mask=reduce(
                cv2.add,  # type: ignore
                [
                    outfit.mask
                    for outfit in detections
                    if outfit.mask is not None
                ],
            ),
        )

    def generate_skin_part(
        self, detection_outputs: list[DetectorOutput]
    ) -> list[DetectorOutput]:
        # TODO: This needs a matching algo, change DO to list[DO] then match
        person_detections = self._find_detections_from_label(
            detection_outputs, "YOLOSEG_PERSON"
        )
        outfit_detections = self._find_detections_from_label(
            detection_outputs, "HUMANPARSE_OUTFIT"
        )
        subtraction_detections = self._find_detections_from_label(
            detection_outputs, ("HUMANPARSE_PHONE", "HUMANPARSE_HAIR")
        )

        # Create Subtractive Mask
        subtraction_mask = None
        if subtraction_detections:
            subtraction_mask = self._flatten_masks_of_detections(
                subtraction_detections
            ).mask

        processed_people_masks = [
            cv2.subtract(person.mask, subtraction_mask)  # type: ignore
            if subtraction_mask is not None
            else person.mask
            for person in person_detections
            if person.mask is not None
        ]

        # Fallback if no Outfit is Found
        if not person_detections and not outfit_detections:
            return []

        if person_detections and not outfit_detections:
            return [
                DetectorOutput(
                    origin="Inferred",
                    label=f"{self.prefix}_SKIN",
                    score=person.score,
                    mask=person_mask,  #  type: ignore
                    bbox=person.bbox,
                    output_type=OutputType.PART,
                )
                for person_mask, person in zip(
                    processed_people_masks,
                    person_detections,
                    strict=False,
                )
            ]

        return [
            DetectorOutput(
                origin="Inferred",
                label=f"{self.prefix}_SKIN",
                score=person.score * outfit.score,
                mask=AdvancedMaskStrategy.skin_mask(person_mask, outfit.mask),  #  type: ignore
                bbox=person.bbox,
                output_type=OutputType.PART,
            )
            for person_mask, person, outfit in zip(
                processed_people_masks,
                person_detections,
                outfit_detections,
                strict=False,
            )
        ]
