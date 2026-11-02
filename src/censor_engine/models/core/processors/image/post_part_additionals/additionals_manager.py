from censor_engine.models.core.detection_part.detection_part import PartList
from censor_engine.models.libraries.configs.config import Config

from ._depth import DepthManager


class AdditionalsManager:
    depth_manager: DepthManager = DepthManager()

    def run_additionals_pipeline(
        self,
        parts_list: PartList,
        config: Config,
    ) -> PartList:
        return self.depth_manager.apply_depth_layers(parts_list, config)
