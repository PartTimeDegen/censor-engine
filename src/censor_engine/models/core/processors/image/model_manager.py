from dataclasses import dataclass, field

from censor_engine.libraries.registries import AIModelRegistry
from censor_engine.models.libraries.ai_models.ai_models import AIModel
from censor_engine.models.libraries.configs.config import Config
from censor_engine.models.libraries.configs.settings._engine_settings import (
    _ExtraAISettings,
    _ProcessingSettings,
)


class ModelManagementStrategy:
    @staticmethod
    def skin_segmentation(models_used: set[str], subtraction_models: set[str]):
        models_used |= {"YoloSeg", "HumanParse"}
        subtraction_models |= {"YoloSeg", "HumanParse"}

    @staticmethod
    def body_segmentation(models_used: set[str], subtraction_models: set[str]):
        models_used |= {"YoloSeg"}
        subtraction_models.discard("YoloSeg")

    @staticmethod
    def focused_roi(models_used: set[str], subtraction_models: set[str]):
        models_used |= {"YoloSeg"}
        subtraction_models.discard("YoloSeg")

    @staticmethod
    def depth_map(models_used: set[str]):
        models_used |= {"YoloDepth", "YoloSeg"}

    @staticmethod
    def clothes_segmentation(
        models_used: set[str], subtraction_models: set[str]
    ):
        models_used |= {"HumanParse"}
        subtraction_models.discard("HumanParse")


@dataclass(slots=True)
class ModelManager:
    config: Config

    extra_settings: _ExtraAISettings = field(init=False)
    processing_settings: _ProcessingSettings = field(init=False)

    models_used_names: set[str] = field(default_factory=set)
    subtraction_models: set[str] = field(default_factory=set)

    running_models: list[AIModel] = field(default_factory=list)

    def __post_init__(self):
        # Set NudeNet as the Minimum
        enabled_parts = self.config.detection.enabled_parts
        if enabled_parts is None or any(
            label.startswith("NUDENET") for label in enabled_parts
        ):
            self.models_used_names = {"NudeNet"}

        # Config Settings
        engine_settings = self.config.engine_settings
        self.extra_settings = engine_settings.extra_ai_model_features
        self.processing_settings = engine_settings.processing

        # Initiate Models
        self._initiate_models()

    def _handle_models_to_load(
        self,
        extra_settings: _ExtraAISettings,
        models_used: set[str],
        subtract_models: set[str],
    ) -> None:
        if extra_settings.skin_segmentation:
            ModelManagementStrategy.skin_segmentation(
                models_used, subtract_models
            )

        if extra_settings.body_segmentation:
            ModelManagementStrategy.body_segmentation(
                models_used, subtract_models
            )
        if extra_settings.focused_roi:
            ModelManagementStrategy.focused_roi(models_used, subtract_models)

        if extra_settings.depth:
            ModelManagementStrategy.depth_map(models_used)

        if extra_settings.clothes_segmentation:
            ModelManagementStrategy.clothes_segmentation(
                models_used, subtract_models
            )

    def _initiate_models(self) -> None:
        # Determine Models
        self._handle_models_to_load(
            self.extra_settings,
            self.models_used_names,
            self.subtraction_models,
        )

        # Handle Enabled Parts
        parts = self.config.detection.enabled_parts
        if parts is not None:
            self.models_used_names |= {part.split("_")[0] for part in parts}
            self.models_used_names.discard("INFERRED")

        self.models_used_names = {
            model.upper() for model in self.models_used_names
        }

        # Initiate the Models
        found_models = AIModelRegistry.get_all()
        for model in self.models_used_names:
            found = found_models[model]()
            found.initiate_model()  # type: ignore
            self.running_models.append(found)  # type: ignore
