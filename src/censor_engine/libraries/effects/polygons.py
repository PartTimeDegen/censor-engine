import numpy as np
from scipy.spatial import Voronoi  # type: ignore

from censor_engine._typing import Image
from censor_engine.libraries.effects._default_args import (
    OUTLINE_COLOUR,
    OUTLINE_WIDTH,
    POINT_DENSITY,
    SEED,
    SIZE,
)
from censor_engine.libraries.registries import EffectRegistry
from censor_engine.models.libraries.effects.effects import PolygonEffect
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from censor_engine.structs.colours import TypeColour


@EffectRegistry.register()
class TriangleGrid(PolygonEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        size: int = SIZE,
        point_density: int = POINT_DENSITY,
        outline_width: int = OUTLINE_WIDTH,
        outline_colour: TypeColour = OUTLINE_COLOUR,
        seed: int = SEED,
    ) -> Image:
        # Transfer Variable
        self.size = size

        return self.helper.generate_polygons(
            effect_context,
            outline_width,
            outline_colour,
            self._polygon_function,
        )

    def _polygon_function(self, width: int, height: int) -> list[np.ndarray]:  # type: ignore
        # Helper Variables
        size = self.size

        # Sized Shape Arrays
        polygon_templates = (
            self.helper.shape_templates.TRIANGLE_A * size,
            self.helper.shape_templates.TRIANGLE_B * size,
        )

        # Get Polygons
        return [
            polygon + np.array((x, y), dtype=np.int32)
            for y in range(0, height, size)
            for x in range(0, width, size)
            for polygon in polygon_templates
        ]


@EffectRegistry.register()
class RandomTriangleGrid(PolygonEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        size: int = SIZE,
        outline_width: int = OUTLINE_WIDTH,
        outline_colour: TypeColour = OUTLINE_COLOUR,
        seed: int = SEED,
    ) -> Image:
        # Transfer Variable
        self.size = size
        self.rng = np.random.default_rng(seed)

        return self.helper.generate_polygons(
            effect_context,
            outline_width,
            outline_colour,
            self._polygon_function,
        )

    def _polygon_function(  # type: ignore
        self,
        width: int,
        height: int,
    ) -> list[np.ndarray]:

        size = self.size

        polygon_template_pairs = (
            (
                self.helper.shape_templates.TRIANGLE_A * size,
                self.helper.shape_templates.TRIANGLE_B * size,
            ),
            (
                self.helper.shape_templates.TRIANGLE_C * size,
                self.helper.shape_templates.TRIANGLE_D * size,
            ),
        )

        return [
            poly
            for y in range(0, height, size)
            for x in range(0, width, size)
            for poly in polygon_template_pairs[self.rng.random() > 0.5]  # noqa: PLR2004
            + np.array([x, y], dtype=np.int32)
        ]


@EffectRegistry.register()
class LloydCrystallise(PolygonEffect):
    def generate_effect(  # type: ignore
        self,
        effect_context: EffectContext,
        *,
        point_density: int = POINT_DENSITY,
        outline_width: int = OUTLINE_WIDTH,
        outline_colour: TypeColour = OUTLINE_COLOUR,
        seed: int = SEED,
        iterations: int = 3,
    ) -> Image:
        # Transfer Variable
        self.point_density = point_density
        self.rng = np.random.default_rng(seed)
        self.iterations = iterations

        return self.helper.generate_polygons(
            effect_context,
            outline_width,
            outline_colour,
            self._polygon_function,
        )

    def _polygon_function(  # type:ignore
        self,
        width: int,
        height: int,
    ) -> list[np.ndarray]:

        # Get Random Starting Points
        rdm_pts = self.rng.random((self.point_density, 2), dtype=np.float32)
        rdm_pts[:, 0] *= width
        rdm_pts[:, 1] *= height

        # Iterations
        relaxed = []
        for _ in range(self.iterations):
            vor = Voronoi(rdm_pts)
            regions = [
                vor.regions[region_index] for region_index in vor.point_region
            ]
            relaxed = [
                vor.vertices[region].mean(axis=0)
                for region in regions
                if region and -1 not in region
            ]

            if not relaxed:
                break

        # Make Polygons
        vor = Voronoi(np.asarray(relaxed, dtype=np.float32))
        return [
            np.asarray(
                [vor.vertices[index] for index in vor.regions[region_index]],
                dtype=np.int32,
            )
            for region_index in vor.point_region
            if vor.regions[region_index]
            and -1 not in vor.regions[region_index]
        ]
