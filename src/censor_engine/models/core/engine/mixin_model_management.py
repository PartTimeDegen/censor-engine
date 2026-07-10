from censor_engine.models.config.base import Config
from censor_engine.models.lib_models.detectors.ai_models import AIModel

from censor_engine.libraries.registries import AIModelRegistry
from censor_engine.structs import Mixin


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

        label_models: set[str] = set().union(*filtered_detectors.values())
        return label_models

    def __select_enabled_models(
        self,
        config: Config,
        detections_enabled: list[str] | str,
        potential_detectors: dict[str, set[str]],
    ) -> set[str]:
        # Core Label Models
        label_models = self.__find_label_detectors(
            potential_detectors,
            detections_enabled,
        )

        # Extras
        extra_settings = config.ai_settings.extras
        extra_features: dict[str, bool] = {
            "_depth": extra_settings.layers != 0,
            "_body_segmentation": extra_settings.body_segmentation,
            "_clothes_segmentation": extra_settings.clothes_segmentation,
            "_roi_focus": extra_settings.focused_roi,
        }

        enabled_extra_features = {
            feature for feature, cond in extra_features.items() if cond
        }

        # Iterate Extras to Get Additional Models
        extra_models: set[str] = set()
        for feature in enabled_extra_features:
            # Error if Missing Model
            model = potential_detectors.get(feature)
            if model is None:
                name = feature[1:].title().replace("_", " ")
                msg = f"Missing Model for {name}"
                raise KeyError(msg)

            extra_models |= model

        return label_models | extra_models

    def __initiate_enabled_models(
        self,
        full_detectors: list,
        enabled_models: set[str],
    ) -> list:
        filtered_detectors = [
            detector()
            for detector in full_detectors
            if detector.model_name in enabled_models
        ]

        [
            detector.model_object.initiate_model()
            for detector in filtered_detectors
        ]

        return filtered_detectors

    def _activate_used_models(self, config: Config) -> list[AIModel]:
        # Bootstrap
        full_detectors = list(AIModelRegistry.get_all().values())
        detections_enabled = config.ai_settings.detections_enabled

        # Extract Potential Detectors for Ease
        potential_detectors: dict[str, set[str]] = {}
        for detector in full_detectors:
            for label in detector.model_classifiers:  # type: ignore
                potential_detectors.setdefault(label, set()).add(
                    detector.model_name  # type: ignore
                )

        # Get Enabled Models
        enabled_models = self.__select_enabled_models(
            config,
            detections_enabled,
            potential_detectors,
        )
        return self.__initiate_enabled_models(full_detectors, enabled_models)
