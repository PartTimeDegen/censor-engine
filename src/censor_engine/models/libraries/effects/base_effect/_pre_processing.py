from censor_engine.models.libraries.effects.base_effect._common_methods import (  # type: ignore
    CommonProcessingMethods,
)
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)


class PreProcessingPipeline(CommonProcessingMethods):
    def pre_process_image(
        self,
        effect_context: EffectContext,
        **kwargs: dict,
    ):
        settings = effect_context.general_settings.pre_processing
        image = effect_context.image
        if settings.greyscale:
            image = self._create_greyscale(effect_context, **kwargs)

        return image
