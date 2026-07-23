from censor_engine.libraries.registries import MaskRegistry
from censor_engine.models.libraries.masks.enums import MaskType


def get_masks(mask_type: MaskType) -> list:
    return [
        mask
        for mask in MaskRegistry.get_all().values()
        if mask.mask_type == mask_type
    ]
