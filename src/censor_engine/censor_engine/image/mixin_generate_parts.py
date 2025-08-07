from uuid import UUID

from censor_engine.detected_part import Part
from censor_engine.models.config import Config
from censor_engine.models.enums import ShapeType
from censor_engine.models.lib_models.detectors import DetectedPartSchema
from censor_engine.models.structs import Mixin


class MixinGenerateParts(Mixin):
    def _create_parts(
        self,
        config: Config,
        file_uuid: UUID,
        detected_parts: list[DetectedPartSchema],
        shape: tuple[int, int, int],
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
                return None

            return Part(
                part_name=detect_part.label,
                part_id=detect_part.part_id,
                score=detect_part.score,
                relative_box=detect_part.relative_box,
                config=config,
                file_uuid=file_uuid,
                image_shape=shape,
            )

        return [
            part for part in map(add_parts, detected_parts) if part is not None
        ]

    def _merge_parts_if_in_merge_groups(self, parts: list[Part]) -> list[Part]:
        def merge_fellow_parts(
            part: Part,
            start_index: int,
            parts: list[Part],
            merged_indices: set[int],
        ) -> Part:
            for index in range(start_index + 1, len(parts)):
                other_part = parts[index]
                if index in merged_indices:
                    continue
                if other_part.get_name() not in part.merge_group:
                    continue

                part.base_masks.extend(other_part.base_masks)
                merged_indices.add(index)

            return part

        # Prep Stuff
        new_parts: list[Part] = []
        merged_indices: set[int] = set()
        for index, part in enumerate(parts):
            if index in merged_indices:
                continue

            if part.merge_group:
                part = merge_fellow_parts(part, index, parts, merged_indices)

                part.compile_base_masks()

            new_parts.append(part)

        return new_parts

    def _apply_and_generate_mask_shapes(
        self,
        parts: list[Part],
    ) -> list[Part]:
        new_parts = []
        for part in parts:
            empty_mask = Part.create_empty_mask(part.image_shape)
            # For Simple Shapes
            if not part.is_merged:
                shape_single = Part.get_shape_class(
                    part.shape_object.single_shape
                )
                part.mask = shape_single.generate(part, empty_mask)
                new_parts.append(part)
                continue

            # For Advanced Shapes
            match part.shape_object.shape_type:
                case ShapeType.BASIC:
                    pass
                case ShapeType.JOINT:
                    part.mask = part.shape_object.generate(part, empty_mask)
                    pass

                case ShapeType.BAR:
                    if not part.is_merged:
                        # Make Basic Shape
                        shape_single = Part.get_shape_class("ellipse")
                        part.mask = shape_single.generate(part, empty_mask)

                    # Make Shape Joint for Bar Basis
                    shape_joint = Part.get_shape_class("joint_ellipse")
                    part.mask = shape_joint.generate(part, empty_mask)

                    # Generate Bar
                    part.mask = part.shape_object.generate(part, empty_mask)

            new_parts.append(part)

        return new_parts
