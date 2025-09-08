class Colour:
    value: tuple[int, int, int]  # Value

    def __repr__(self) -> str:
        return f"{self.value} ({self.get_colour(self.value)})"

    def __str__(self) -> str:
        return f"{self.value}"

    def __init__(
        self,
        colour_name_or_rgb_value: str | tuple[int, int, int] = "WHITE",
        *,
        already_bgr: bool = False,
    ) -> None:
        if already_bgr and isinstance(
            colour_name_or_rgb_value,
            tuple,
        ):
            self.value = colour_name_or_rgb_value

        elif isinstance(colour_name_or_rgb_value, tuple):
            self.value = self._flip_colour(colour_name_or_rgb_value)

        elif isinstance((colour_name_or_rgb_value), str):
            self.value = self._flip_colour(_colours[colour_name_or_rgb_value])

        else:
            msg = "Bad Colour:"
            raise ValueError(msg, colour_name_or_rgb_value)

    def _flip_colour(
        self,
        colour: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        return (colour[2], colour[1], colour[0])

    def get_colour(self, colour: tuple[int, int, int]) -> str:
        flipped_value = self._flip_colour(colour)
        for name, val in _colours.items():
            if flipped_value == val:
                return name

        msg = f"Missing Colour: {colour}"
        raise ValueError(msg)


_colours = {
    # Base Colours
    "BLACK": (0, 0, 0),
    "GREY": (128, 128, 128),
    "WHITE": (255, 255, 255),
    "RED": (255, 0, 0),
    "PINK": (255, 192, 203),
    "BLUE": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "YELLOW": (255, 255, 0),
    "ORANGE": (255, 165, 0),
    "PURPLE": (160, 32, 240),
    "BROWN": (123, 63, 0),  # Chocolate
    # Hot Colours
    "HOT_RED": (189, 24, 22),
    "HOT_PINK": (255, 16, 240),  # Technically "Neon Pink"
    "HOT_BLUE": (0, 255, 255),  # Aqua Blue
    "HOT_GREEN": (173, 255, 47),
    "HOT_YELLOW": (255, 215, 0),  # Gold
    "HOT_ORANGE": (255, 117, 24),
    "HOT_PURPLE": (127, 0, 255),  # Violet
    "HOT_BROWN": (139, 0, 0),
    # Baby Colours
    "BABY_RED": (255, 119, 121),
    "BABY_PINK": (242, 172, 185),
    "BABY_BLUE": (137, 207, 240),
    "BABY_GREEN": (140, 255, 158),
    "BABY_YELLOW": (255, 241, 215),
    "BABY_ORANGE": (255, 165, 0),
    "BABY_PURPLE": (202, 155, 247),
    "BABY_BROWN": (197, 167, 122),
    # Light Colours
    "LIGHT_RED": (250, 160, 160),
    "LIGHT_PINK": (255, 182, 193),
    "LIGHT_BLUE": (173, 216, 230),
    "LIGHT_GREEN": (152, 251, 152),  # Mint Green
    "LIGHT_YELLOW": (255, 250, 160),  # Pastel Yellow
    "LIGHT_ORANGE": (255, 127, 80),  # Coral
    "LIGHT_PURPLE": (203, 195, 227),
    "LIGHT_BROWN": (193, 154, 107),
    # Dark Colours
    "DARK_RED": (128, 0, 0),  # Maroon
    "DARK_PINK": (170, 51, 106),
    "DARK_BLUE": (0, 0, 128),  # Navy Blue
    "DARK_GREEN": (2, 48, 52),
    "DARK_YELLOW": (253, 218, 13),  # Cadmium Yellow
    "DARK_ORANGE": (255, 95, 21),
    "DARK_PURPLE": (128, 0, 128),
    "DARK_BROWN": (92, 64, 51),
    # Unique Colours
    "TURQUOISE": (0, 128, 128),
    "AZURE": (0, 127, 255),
    "COTTON_CANDY": (255, 188, 217),
    "CHERRY": (255, 183, 197),
    "MATTE_BLACK": (40, 40, 43),
    "ONYX": (53, 57, 53),
    "DARK_TAN": (152, 133, 88),
    "TAN": (210, 180, 140),
    "TUSCAN_RED": (124, 48, 48),
    "WINE": (114, 47, 55),
    "ARMY_GREEN": (69, 75, 27),
    "CRIMSON": (220, 20, 60),
    "AMARANTH": (159, 43, 104),
    "BURGUNDY": (128, 0, 32),
    "BYZANTIUM": (112, 41, 99),
    "IRIS": (93, 63, 211),
    "NEON_RED": (255, 49, 49),
    "TYRIAN_PURPLE": (99, 3, 48),
    "VENETIAN_RED": (164, 42, 4),
}
