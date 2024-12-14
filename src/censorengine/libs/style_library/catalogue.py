from censorengine.libs.style_library.styles.core import (
    blur,
    box,
    colour,
    transparent,
    dev,
    text,
)


style_catalogue = {
    **blur.effects,
    **box.effects,
    **colour.effects,
    **transparent.effects,
    **dev.effects,
    # **text.effects,
}  # type: ignore
