from typing import Literal

import cv2
import numpy as np

from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from censor_engine.structs.colours import Colour, TypeColour


class _PointDistributions:
    @staticmethod
    def random(
        rng: np.random.Generator,
        width: int,
        height: int,
        point_density: int,
    ):
        # This has random points on the image.
        return np.column_stack(
            (
                rng.integers(0, width, point_density),
                rng.integers(0, height, point_density),
            )
        )

    @staticmethod
    def grid(
        rng: np.random.Generator,
        width: int,
        height: int,
        point_density: int,
        jitter: float,
    ):
        # This puts the points on a grid then has settings to "jitter" it.
        grid = int(np.sqrt(point_density))
        xs = np.linspace(0, width - 1, grid)
        ys = np.linspace(0, height - 1, grid)

        xv, yv = np.meshgrid(xs, ys)

        points = np.column_stack((xv.ravel(), yv.ravel()))

        dx = width / grid * jitter
        dy = height / grid * jitter

        points += rng.uniform(
            low=(-dx, -dy),
            high=(dx, dy),
            size=points.shape,
        )  # type: ignore

        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        return points

    @staticmethod
    def poisson(
        rng: np.random.Generator,
        width: int,
        height: int,
        point_density: int,
    ):
        # This one is similar to Random but gives better clustering
        return np.column_stack(
            (
                rng.integers(0, width, point_density),
                rng.integers(0, height, point_density),
            )
        )


class CrystallisationHelpers:
    def _generate_points(
        self,
        width: int,
        height: int,
        point_density: int,
        distribution: str,
        jitter: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        args = [rng, width, height, point_density]

        match distribution:
            case "random":
                points = _PointDistributions.random(*args)  # type: ignore
            case "grid":
                args += [jitter]
                points = _PointDistributions.grid(*args)  # type: ignore
            case "poisson":
                points = _PointDistributions.poisson(*args)  # type: ignore
            case _:
                msg = f"Unknown point distribution: {distribution}"
                raise ValueError(msg)

        return points.astype(np.float32)

    def _triangulate(
        self,
        points: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        subdiv = cv2.Subdiv2D((0, 0, width, height))  # type: ignore

        for point in points:
            subdiv.insert(tuple(point))

        return subdiv.getTriangleList().astype(np.int32)  # type: ignore

    def _fill_triangles(
        self,
        output: np.ndarray,
        image: np.ndarray,
        triangle: np.ndarray,
        height: int,
        width: int,
        outline_colour: Colour,
        outline_width: int,
    ) -> None:
        vertices = triangle.reshape(3, 2)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, vertices, 1)  # type: ignore

        colour = cv2.mean(image, mask=mask)[:3]

        cv2.fillConvexPoly(output, vertices, colour)  # type: ignore

        if outline_width > 0:
            cv2.polylines(
                output,  # type: ignore
                [vertices],
                isClosed=True,
                color=outline_colour.value,
                thickness=outline_width,
            )

    def generate_crystals(
        self,
        effect_context: EffectContext,
        point_density: int,
        outline_width: int,
        outline_colour: TypeColour,
        seed: int,
        point_distribution: Literal["random", "grid", "poisson"],
        *,
        jitter: float = 0.0,
    ):
        # Get Variables
        image = effect_context.image
        height, width = effect_context.image_shape
        colour_obj = Colour(outline_colour)

        # Get RNG Object
        rng = np.random.default_rng(seed)

        # Generate Base Points
        points = self._generate_points(
            width=width,
            height=height,
            point_density=point_density,
            distribution=point_distribution,
            jitter=jitter,
            rng=rng,
        )

        # Handle the Corners
        corners = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [0, height - 1],
                [width - 1, height - 1],
            ],
            dtype=np.float32,
        )
        points = np.vstack((points, corners))

        # Triangulate Points
        triangles = self._triangulate(points, width, height)

        # Fill Triangles
        result = np.zeros_like(image)
        for triangle in triangles:
            self._fill_triangles(
                output=result,
                image=image,
                triangle=triangle,
                height=height,
                width=width,
                outline_colour=colour_obj,
                outline_width=outline_width,
            )

        return result
