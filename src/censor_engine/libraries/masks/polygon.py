from censor_engine._typing import MaskImage
from censor_engine.libraries.registries import MaskRegistry
from censor_engine.models.libraries.masks.masks import PolygonMask
from censor_engine.models.libraries.masks.schemas import MaskContext


@MaskRegistry.register()
class Star(PolygonMask):
    base_mask = "Star"
    single_mask = "Star"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...


@MaskRegistry.register()
class Hexagon(PolygonMask):
    base_mask = "Hexagon"
    single_mask = "Hexagon"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...


@MaskRegistry.register()
class Octagon(PolygonMask):
    base_mask = "Octagon"
    single_mask = "Octagon"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...


@MaskRegistry.register()
class SatanicPentagon(PolygonMask):
    base_mask = "SatanicPentagon"
    single_mask = "SatanicPentagon"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...


@MaskRegistry.register()
class Cross(PolygonMask):
    base_mask = "Cross"
    single_mask = "Cross"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...


@MaskRegistry.register()
class Bubble(PolygonMask):
    base_mask = "Bubble"
    single_mask = "Bubble"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...


@MaskRegistry.register()
class Apple(PolygonMask):
    base_mask = "Apple"
    single_mask = "Apple"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...


@MaskRegistry.register()
class Lock(PolygonMask):
    base_mask = "Lock"
    single_mask = "Lock"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...


# Cards
@MaskRegistry.register()
class Heart(PolygonMask):
    base_mask = "Heart"
    single_mask = "Heart"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...


@MaskRegistry.register()
class Diamond(PolygonMask):
    base_mask = "Diamond"
    single_mask = "Diamond"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...


@MaskRegistry.register()
class Club(PolygonMask):
    base_mask = "Club"
    single_mask = "Club"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...


@MaskRegistry.register()
class Spade(PolygonMask):
    base_mask = "Spade"
    single_mask = "Spade"

    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        ...
