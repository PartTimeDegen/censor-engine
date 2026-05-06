from censor_engine.libs.registries import DetectorRegistry
from censor_engine.models.config.base import Config
from censor_engine.models.structs import Mixin


class MixinModelManagement(Mixin):
    def __find_label_detectors(
        self,
        potential_detectors: dict[str, set[str]],
        detections_enabled: list[str] | str,
    ) -> set[str]:

        # Get List of Parts and their Detector

        filtered_detectors = potential_detectors
        if detections_enabled != "all":
            filtered_detectors = {
                k: v
                for k, v in filtered_detectors.items()
                if k in detections_enabled
            }
        label_models: set[str] = set().union(*potential_detectors.values())
        return label_models

    def __select_enabled_models(self):
        pass

    def __initiate_enabled_models(self):
        pass

    def _activate_used_models(self, config: Config):
        # Bootstrap
        full_detectors = list(DetectorRegistry.get_all().values())
        detections_enabled = config.ai_settings.detections_enabled

        # Extract Potential Detectors for Ease
        potential_detectors: dict[str, set[str]] = {}
        for detector in full_detectors:
            for label in detector.model_classifiers:  # type: ignore
                potential_detectors.setdefault(label, set()).add(
                    detector.model_name  # type: ignore
                )

        # Core Label Models
        label_models = self.__find_label_detectors(
            potential_detectors,
            detections_enabled,
        )

        # TODO: This can be DRY, Need to figure it out

        # Depth Model
        # TODO: This needs to be implemented
        depth_model: set[str] = set()
        # if config.ai_settings.extras.layers != 0:
        #     if model := potential_detectors.get("_depth"):
        #         depth_model = {model}
        #     else:
        #         msg = "Missing Model for Depth"
        #         raise KeyError(msg)

        # Body Seg Model
        # TODO: This needs to be implemented
        body_seg_model: set[str] = set()
        if config.ai_settings.extras.body_segmentation:
            if model := potential_detectors.get("_body_segmentation"):
                body_seg_model = model
            else:
                msg = "Missing Model for Body Segmentation"
                raise KeyError(msg)

        # Body Seg Model
        # TODO: This needs to be implemented
        clothes_seg_model: set[str] = set()
        # if config.ai_settings.extras.clothes_segmentation:
        #     if model := potential_detectors.get("_clothes_segmentation"):
        #         clothes_seg_model = {model}
        #     else:
        #         msg = "Missing Model for Clothes Segmentation"
        #         raise KeyError(msg)

        # Focused ROI Model
        # TODO: This needs to be implemented
        roi_focus: set[str] = set()
        # if config.ai_settings.extras.focused_roi:
        #     if model := potential_detectors.get("_roi_focus"):
        #         roi_focus = {model}
        #     else:
        #         msg = "Missing Model for ROI Focus"
        #         raise KeyError(msg)

        full_models = (
            label_models
            | depth_model
            | body_seg_model
            | clothes_seg_model
            | roi_focus
        )
        print()
        print(full_models)

        exit()
