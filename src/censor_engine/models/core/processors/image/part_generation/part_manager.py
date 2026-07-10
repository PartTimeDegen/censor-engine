from uuid import UUID

from censor_engine._typing import ImageShape
from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.core.path_manager.path_manager import PathManager
from censor_engine.models.libraries.detectors.schemas import DetectorOutput

from ._merge_mechanism import MergeMechanismManager
from ._merge_on_part_state import PartStateMechanism


class PartManager:
    merge_mechanism: MergeMechanismManager = MergeMechanismManager()
    state_mechanism: PartStateMechanism = PartStateMechanism()

    def _create_parts_objects_from_detector_outputs(
        self,
        path_manager: PathManager,
        detected_outputs: list[DetectorOutput],
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
            path_manager: The system Path Manager
            detected_outputs: List of the detected parts from the detectors
            file_uuid: The file UUID, used for bars to maintain angle
            image_shape: Image Shape

        Returns:
            List of parts as Part objs

        """
        detection_config = path_manager.config.detection

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

            if detector_output.label not in detection_config.enabled_parts:
                return None

            part_config = detection_config.parts[detector_output.label]
            if detector_output.score < part_config.minimum_score:
                return None

            return Part(
                detector_output=detector_output,
                config=path_manager.config,
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

    def run_part_generation_pipeline(
        self,
        path_manager: PathManager,
        detected_outputs: list[DetectorOutput],
        file_uuid: UUID,
        image_shape: ImageShape,
    ):
        # Generate Part Objects
        parts = self._create_parts_objects_from_detector_outputs(
            path_manager,
            detected_outputs,
            file_uuid,
            image_shape,
        )

        # Sort Parts
        parts = self._sort_parts(parts)

        # Run Mechanisms to Merge Parts
        merged_parts = self.merge_mechanism.merge_parts_based_on_merge_method()
        processed_parts = (
            self.state_mechanism.handle_mask_overlaps_based_on_part_state()
        )
