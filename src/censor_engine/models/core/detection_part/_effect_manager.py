from dataclasses import dataclass
from functools import reduce

from censor_engine._typing import Image, MaskImage
from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.core.detection_part._part_properties import (
    PartProperties,
)
from censor_engine.models.libraries.configs._helper_types import ListOfCensors
from censor_engine.models.libraries.effects.base_effect.base_effect import (
    Effect,
)
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from censor_engine.structs.censors import Censor


@dataclass(slots=True)
class EffectManager:
    list_of_censors: ListOfCensors
    list_of_reverse_censors: ListOfCensors

    def _apply_censor_effect(
        self,
        ec: EffectContext,
        censor: Censor,
    ) -> EffectContext:
        """
        This method is used via the reduce function to handle adding the
        censor effects to the input image.

        The method contains the whole pipeline for applying censor effects to
        the image.

        TODO: When the other parts of the pipeline get made, update the params.

        Args:
            ec: effect context
            censor: censor applied

        Returns:
            effect context with the image updated

        """
        # Get he effect class and shorthand the params
        effect = self.get_effect_class(censor.effect)
        params = censor.parameters

        # Effect Pipeline
        ec.image = effect.generate_effect_pre_process(ec, **params)
        ec.image = effect.generate_effect(ec, **params)
        ec.image = effect.generate_effect_post_process(ec, **params)

        # Apply the Effect
        ec.image = effect.apply_effect_to_image(ec)

        return ec

    def apply_effects_from_list_of_reverse_censors(
        self,
        input_image: Image,
        inverse_full_mask: MaskImage,
        part_properties: PartProperties,
    ) -> Image:

        effect_context = EffectContext(
            image=input_image,
            mask=inverse_full_mask,
            part_properties=part_properties,
        )

        effect_context = reduce(
            self._apply_censor_effect,
            self.list_of_reverse_censors,
            effect_context,
        )

        return effect_context.image

    def apply_effects_from_list_of_censors(
        self,
        input_image: Image,
        mask: MaskImage,
        part_properties: PartProperties,
    ) -> Image:
        effect_context = EffectContext(
            image=input_image,
            mask=mask,
            part_properties=part_properties,
        )

        effect_context = reduce(
            self._apply_censor_effect,
            self.list_of_censors,
            effect_context,
        )

        return effect_context.image

    @staticmethod
    def get_effect_class(effect: str) -> Effect:
        effects = EffectRegistry.get_all()
        if effect not in effects:
            msg = f"Effect {effect} does not Exist! {[effects.keys()]}"
            raise ValueError(msg)

        return effects[effect]()
