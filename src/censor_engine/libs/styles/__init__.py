from censor_engine.libs.style_library.styles import (
    blur,
    box,
    colour,
    pixelate,
    transparent,
    dev,
    text,
    edge_detection,
    noise,
    stylisation,
)

style_catalogue = {
    **blur.effects,
    **box.effects,
    **colour.effects,
    **dev.effects,
    **edge_detection.effects,
    **noise.effects,
    **pixelate.effects,
    **stylisation.effects,
    # **text.effects,
    **transparent.effects,
}  # type: ignore
