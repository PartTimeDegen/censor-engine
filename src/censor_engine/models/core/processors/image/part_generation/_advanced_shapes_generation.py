import cv2

from censor_engine.models.core.detection_part._schemas import PartNameType
from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.libraries.masks.enums import MaskType
from censor_engine.models.libraries.masks.schemas import MaskContext


class MaskConversionStrategy:
    @staticmethod
    def handle_blanket_masks(part: Part, mask_context: MaskContext) -> Part:
        part.mask_manager.current_mask = (
            part.mask_manager.mask_obj.generate_mask(mask_context)
        )
        return part

    @staticmethod
    def handle_simple_masks(part: Part, mask_context: MaskContext) -> Part:
        part.mask_manager.current_mask = (
            part.mask_manager.mask_obj.generate_single_mask(mask_context)
        )
        return part

    @staticmethod
    def handle_advanced_masks(part: Part, mask_context: MaskContext) -> Part:
        part.mask_manager.current_mask = (
            part.mask_manager.mask_obj.generate_mask(mask_context)
        )
        return part

    @staticmethod
    def handle_segment_masks(part: Part) -> Part:
        segment_mask = part.detector_output.mask
        if segment_mask is None:
            msg = "Part was given MASK suffix when part doesn't contain a mask"
            raise TypeError(msg)
        part.mask_manager.current_mask = segment_mask
        return part


class AdvancedShapeMaskGenerator:
    def convert_masks_to_advanced_versions(
        self, parts: list[Part]
    ) -> list[Part]:
        if len(parts) == 0:
            return []

        processed_parts_list: list[Part] = []
        for part in parts:
            if part.get_name(PartNameType.NAME).endswith("_MASK"):
                MaskConversionStrategy.handle_segment_masks(part)
                processed_parts_list.append(part)
                continue

            # Prepare Context
            empty_mask = part.mask_manager.create_empty_mask()
            mask_context = MaskContext(
                part_name=part.mask_manager.mask_name,
                part_properties=part.properties,
                mask=part.mask_manager.current_mask,
                base_empty_mask=empty_mask,
                file_uuid=part.file_uuid,
            )

            # Handle Conversions
            if part.mask_manager.mask_obj.mask_type == MaskType.BLANKET:
                MaskConversionStrategy.handle_blanket_masks(part, mask_context)
            elif not part.properties.is_merged:
                MaskConversionStrategy.handle_simple_masks(part, mask_context)
            else:
                MaskConversionStrategy.handle_advanced_masks(
                    part, mask_context
                )
            processed_parts_list.append(part)

        return processed_parts_list
