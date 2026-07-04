from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.enums import MergeMethod


class PartStateMechanism:
    def handle_mask_overlaps_based_on_part_state(self): ...
