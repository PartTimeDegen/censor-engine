from censorengine.libs.style_library.styles.core import (
    blur,
    box,
    colour,
    transparent,
    dev,
    text,
    edge_detection,
    noise,
)


style_catalogue = {
    **blur.effects,
    **box.effects,
    **colour.effects,
    **transparent.effects,
    **dev.effects,
    # **text.effects,
    **edge_detection.effects,
    **noise.effects,
}  # type: ignore
