from functools import reduce
from uuid import UUID

import cv2

from censor_engine.models.core.detection_part._schemas import PartNameType
from censor_engine.models.core.detection_part.detection_part import (
    Part,
    PartList,
)
from censor_engine.models.libraries.ai_models._enums import OutputType
from censor_engine.models.libraries.ai_models.output_schemas import (
    DetectorOutput,
)
from censor_engine.models.libraries.configs.config import Config
from censor_engine.typing import ImageShape

from ._advanced_shapes_generation import (
    AdvancedShapeMaskGenerator,
)
from ._merge_mechanism import MergeMechanismManager
from ._state_mechanism import PartStateMechanism


class PartManager:
    merge_mechanism: MergeMechanismManager = MergeMechanismManager()
    state_mechanism: PartStateMechanism = PartStateMechanism()
    mask_generator: AdvancedShapeMaskGenerator = AdvancedShapeMaskGenerator()

    def _create_parts_objects_from_detector_outputs(
        self,
        detected_outputs: list[DetectorOutput],
        config: Config,
        file_uuid: UUID,
        image_shape: ImageShape,
    ):
        """
        This function creates the list of Parts for CensorEngine to keep track
        of.

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

        Args:
            config: Found config file
            detected_outputs: List of the detected parts from the detectors
            file_uuid: The file UUID, used for bars to maintain angle
            image_shape: Image Shape

        Returns:
            List of parts as Part objs

        """
        detection_config = config.detection

        # Map and Filter Parts for Missing Information
        def add_parts(detector_output: DetectorOutput) -> Part | None:
            """
            Generates the parts using the Part constructor. Also checks that
            the part is in the enabled parts.

            The structure could be improved but the reason I've used a map()
            instead of list comprehensions is because it's faster.

            Args:
                detector_output: Detector Output

            Returns:
                The Part object from it, or None if it's invalid

            """
            if detector_output.label is None:
                return None

            if (
                detection_config.enabled_parts is not None
                and detector_output.label not in detection_config.enabled_parts
            ):
                return None

            part_config = detection_config.part_settings[detector_output.label]
            if detector_output.score < part_config.minimum_score:
                return None

            return Part(
                detector_output=detector_output,
                config=config,
                file_uuid=file_uuid,
                image_shape=image_shape,
            )

        return [
            part
            for part in map(add_parts, detected_outputs)
            if part is not None
        ]

    def _sort_parts(self, parts: list[Part]) -> list[Part]:
        return sorted(
            parts,
            key=lambda x: (x.properties.settings.state, x.get_name()),
            reverse=True,
        )

    def _process_part_parts(
        self,
        list_of_parts: list[DetectorOutput],
        config: Config,
        file_uuid: UUID,
        image_shape: ImageShape,
    ):
        # Generate Part Objects
        parts = self._create_parts_objects_from_detector_outputs(
            list_of_parts,
            config,
            file_uuid,
            image_shape,
        )

        # Sort Parts
        sorted_parts = self._sort_parts(parts)

        # Run Mechanisms to Merge Parts by Method
        merged_parts = self.merge_mechanism.merge_parts_based_on_merge_method(
            parts=sorted_parts,
            merge_method=config.image.merging.method,
            groups=config.groups.merging,
        )

        # Convert Masks to Advanced Forms
        processed_parts = (
            self.mask_generator.convert_masks_to_advanced_versions(
                merged_parts
            )
        )

        # Compile Masks Layers into One
        for part in processed_parts:
            if part.get_name(PartNameType.NAME).endswith("_MASK"):
                continue
            part.mask_manager.compile_base_masks()

        # Handle State Logic for Masks to Prevent Overlap
        cleaned_merged_parts = processed_parts
        if config.engine_settings.extra_ai_model_features.depth is None:
            cleaned_merged_parts = (
                self.state_mechanism.handle_mask_overlaps_based_on_part_state(
                    merged_parts
                )
            )

        return cleaned_merged_parts

    def run_part_generation_pipeline(
        self,
        detected_outputs: list[DetectorOutput],
        config: Config,
        file_uuid: UUID,
        image_shape: ImageShape,
    ) -> PartList:
        # Organise Parts into Types
        list_of_parts = [
            do for do in detected_outputs if do.output_type == OutputType.PART
        ]
        depth_list = [
            do for do in detected_outputs if do.output_type == OutputType.DEPTH
        ]
        depth_output: DetectorOutput | None = (
            depth_list[0] if depth_list else None
        )

        # Process Part Parts
        processed_parts = self._process_part_parts(
            list_of_parts,
            config,
            file_uuid,
            image_shape,
        )

        # Extras
        people_masks = [
            do.mask
            for do in detected_outputs
            if do.label == "YOLOSEG_PERSON"
            if do.mask is not None
        ]
        combined_people_masks = (
            reduce(cv2.add, people_masks) if people_masks else None
        )

        return PartList.model_validate(
            {
                "parts": processed_parts,
                "additionals": {
                    "depth": depth_output,
                    "people_mask": combined_people_masks,
                },
            }
        )
