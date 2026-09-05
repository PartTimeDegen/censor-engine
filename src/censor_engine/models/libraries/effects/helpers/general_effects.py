from censor_engine._typing import Image, ImageShape


class GeneralHelpers:
    def normalise_factor(
        self,
        shape: ImageShape,
        factor: float,
    ) -> float:
        new_factor = min(shape) / max(shape) * factor
        if new_factor < 1:
            new_factor = 1
        elif new_factor % 2 == 0:
            new_factor += 1

        return new_factor

    def apply_factor(
        self,
        image: Image,
        factor: float,
    ) -> tuple[int, int]:
        # Fixing Strength
        factor = factor * 4 + 1

        factor = self.normalise_factor(image, factor)  # type: ignore

        if factor < 1:
            factor = 1
        elif factor % 2 == 0:
            factor += 1

        image_ratio = (max(image.shape) - min(image.shape)) / min(image.shape)

        factor_ratio = factor / image_ratio
        return (
            int(factor_ratio * min(image.shape)),  # Min Factor
            int(factor_ratio * max(image.shape)),  # Max Factor
        )
