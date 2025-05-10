from uuid import UUID
from censorengine.backend.constants.typing import Mask
from censorengine.backend.models.config import Config
from censorengine.backend.models.structures.detected_part import Part
from censorengine.backend.models.structures.enums import ShapeType
from censorengine.models.detectors import DetectedPartSchema


class ImageComponentGenerateParts:
    def _create_parts(
        self,
        config: Config,
        empty_mask: Mask,
        file_uuid: UUID,
        detected_parts: list[DetectedPartSchema],
    ) -> list[Part]:
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

        # Map and Filter Parts for Missing Information
        def add_parts(detect_part: DetectedPartSchema) -> Part | None:
            """
            Generates the parts using the Part constructor. Also checks that
            the part is in the enabled parts.

            The structure could be improved but the reason I've used a map()
            instead of list comprehensions is because it's faster.

            :param DetectedPartSchema detect_part: Output from the Detector class method
            :return Optional[Part]: A Part object (or None)
            """
            if detect_part.label not in config.censor_settings.enabled_parts:
                return

            return Part(
                part_name=detect_part.label,
                score=detect_part.score,
                relative_box=detect_part.relative_box,
                empty_mask=empty_mask,
                config=config,
                file_uuid=file_uuid,
            )

        return [part for part in map(add_parts, detected_parts) if part is not None]

    def _merge_parts_if_in_merge_groups(self, parts: list[Part]) -> list[Part]:
        new_parts = []
        for index, part in enumerate(parts):
            if not part.merge_group:
                continue

            for other_part in parts[index + 1 :]:
                if part == other_part:
                    continue
                if other_part.part_name not in part.merge_group:
                    continue

                part.base_masks.append(other_part.base_masks[0])
                parts.remove(other_part)

            part.compile_base_masks()
            new_parts.append(part)

        return new_parts

    def _apply_and_generate_mask_shapes(
        self, empty_mask: Mask, parts: list[Part]
    ) -> list[Part]:
        new_parts = []
        for part in parts:
            # For Simple Shapes
            if not part.is_merged:
                shape_single = Part.get_shape_class(part.shape_object.single_shape)
                part.mask = shape_single.generate(part, empty_mask.copy())

            # For Advanced Shapes
            match part.shape_object.shape_type:
                case ShapeType.BASIC:
                    pass
                case ShapeType.JOINT:
                    part.mask = part.shape_object.generate(part, empty_mask.copy())

                case ShapeType.BAR:
                    if not part.is_merged:
                        # Make Basic Shape
                        shape_single = Part.get_shape_class("ellipse")
                        part.mask = shape_single.generate(part, empty_mask.copy())

                    # Make Shape Joint for Bar Basis
                    shape_joint = Part.get_shape_class("joint_ellipse")
                    part.mask = shape_joint.generate(part, empty_mask.copy())

                    # Generate Bar
                    part.mask = part.shape_object.generate(part, empty_mask.copy())

            new_parts.append(part)

        return new_parts
