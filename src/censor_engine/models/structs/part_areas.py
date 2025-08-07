# Regions and Such
from dataclasses import dataclass, field


@dataclass(slots=True)
class Coord:
    X: int
    Y: int

    def convert_to_tuple(self) -> tuple[int, int]:
        return (self.X, self.Y)


@dataclass(slots=True)
class AreaDimensions:
    width: int
    height: int


@dataclass(slots=True)
class Region:
    relative_box: tuple[int, int, int, int]  # x, y, width, height
    image_bounds: tuple[int, int]

    # Basic Features
    top_left: Coord = field(init=False)
    bot_right: Coord = field(init=False)
    area_dimensions: AreaDimensions = field(init=False)

    # Useful Info
    centre: Coord = field(init=False)
    radius: tuple[int, int] = field(init=False)

    def __post_init__(self):
        # Basic Features
        self.top_left = Coord(
            self.relative_box[0],
            self.relative_box[1],
        )
        self.area_dimensions = AreaDimensions(
            self.relative_box[2],
            self.relative_box[3],
        )
        self.bot_right = Coord(
            self.top_left.X + self.area_dimensions.width,
            self.top_left.Y + self.area_dimensions.height,
        )

        # Useful Info
        self.centre = Coord(
            self.top_left.X + int(self.area_dimensions.width / 2),
            self.top_left.Y + int(self.area_dimensions.height / 2),
        )
        self.radius = (
            int(self.area_dimensions.width / 2),
            int(self.area_dimensions.height / 2),
        )

    def get_corners(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            self.top_left.convert_to_tuple(),
            self.bot_right.convert_to_tuple(),
        )


@dataclass(slots=True)
class ApproximateRegion:
    region: Region
    approximate_percent_region: float
    image_bounds: tuple[int, int]

    top_left_approx: Region = field(init=False)
    bot_right_approx: Region = field(init=False)

    def __post_init__(self):
        # Get Approximate Area Dimensions
        approximate_dimensions = AreaDimensions(
            int(self.approximate_percent_region * self.region.area_dimensions.width),
            int(self.approximate_percent_region * self.region.area_dimensions.height),
        )

        # Get Regions of Top Left and Bottom Right
        self.top_left_approx = Region(
            (
                self.region.top_left.X - int(approximate_dimensions.width / 2),
                self.region.top_left.Y - int(approximate_dimensions.height / 2),
                approximate_dimensions.width,
                approximate_dimensions.height,
            ),
            self.image_bounds,
        )
        self.bot_right_approx = Region(
            (
                self.region.bot_right.X - int(approximate_dimensions.width / 2),
                self.region.bot_right.Y - int(approximate_dimensions.height / 2),
                approximate_dimensions.width,
                approximate_dimensions.height,
            ),
            self.image_bounds,
        )


@dataclass(slots=True)
class PartArea:
    """
    This is used to manage the areas of parts
    """

    relative_box: tuple[int, int, int, int]  # x, y, width, height
    approximate_percent_region: float  # 0.0+
    image_bounds: tuple[int, int]

    # Generated
    region: Region = field(init=False)

    # # Approximate Region
    approx_region: ApproximateRegion = field(init=False)

    def __post_init__(self):
        if (pc := self.approximate_percent_region) <= 0.0:
            raise ValueError(f"Approximate Percent Region ({pc}) is below 0.0")

        # Precise Region
        self.region = Region(self.relative_box, self.image_bounds)

        # Approximate Region
        self.approx_region = ApproximateRegion(
            self.region,
            self.approximate_percent_region,
            self.image_bounds,
        )

    def check_in_approx_region(self, area: Region) -> bool:
        top_left_check = (
            # Top Left X
            area.top_left.X >= self.approx_region.top_left_approx.top_left.X
            and area.top_left.X <= self.approx_region.top_left_approx.bot_right.X
            # Top Left Y
            and area.top_left.Y >= self.approx_region.top_left_approx.top_left.Y
            and area.top_left.Y <= self.approx_region.top_left_approx.bot_right.Y
        )
        bottom_right_check = (
            # Bottom Right X
            area.bot_right.X >= self.approx_region.bot_right_approx.top_left.X
            and area.bot_right.X <= self.approx_region.bot_right_approx.bot_right.X
            # Bottom Right Y
            and area.bot_right.Y >= self.approx_region.bot_right_approx.top_left.Y
            and area.bot_right.Y <= self.approx_region.bot_right_approx.bot_right.Y
        )
        return top_left_check and bottom_right_check
