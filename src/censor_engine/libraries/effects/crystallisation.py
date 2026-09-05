from censor_engine._typing import Image
from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.libraries.effects.effects import (
    CrystallisationEffect,
)
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from censor_engine.structs.colours import TypeColour

from ._default_args import OUTLINE_COLOUR, OUTLINE_WIDTH, POINT_DENSITY, SEED


@EffectRegistry.register()
class GridCrystallise(CrystallisationEffect):
    def generate_effect(  # type:ignore
        self,
        effect_context: EffectContext,
        *,
        point_density: int = POINT_DENSITY,
        outline_width: int = OUTLINE_WIDTH,
        outline_colour: TypeColour = OUTLINE_COLOUR,
        seed: int = SEED,
        jitter: float = 0.25,
    ) -> Image:
        return self.helper.generate_crystals(
            effect_context=effect_context,
            point_density=point_density,
            outline_width=outline_width,
            outline_colour=outline_colour,
            seed=seed,
            point_distribution="grid",
            jitter=jitter,
        )


@EffectRegistry.register()
class RandomCrystallise(CrystallisationEffect):
    def generate_effect(  # type:ignore
        self,
        effect_context: EffectContext,
        *,
        point_density: int = POINT_DENSITY,
        outline_width: int = OUTLINE_WIDTH,
        outline_colour: TypeColour = OUTLINE_COLOUR,
        seed: int = SEED,
    ) -> Image:
        return self.helper.generate_crystals(
            effect_context=effect_context,
            point_density=point_density,
            outline_width=outline_width,
            outline_colour=outline_colour,
            seed=seed,
            point_distribution="random",
        )


@EffectRegistry.register()
class Crystallise(CrystallisationEffect):
    def generate_effect(  # type:ignore
        self,
        effect_context: EffectContext,
        *,
        point_density: int = POINT_DENSITY,
        outline_width: int = OUTLINE_WIDTH,
        outline_colour: TypeColour = OUTLINE_COLOUR,
        seed: int = SEED,
    ) -> Image:
        return self.helper.generate_crystals(
            effect_context=effect_context,
            point_density=point_density,
            outline_width=outline_width,
            outline_colour=outline_colour,
            seed=seed,
            point_distribution="poisson",
        )
