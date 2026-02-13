from typing import Any

from censor_engine.models.lib_models.detectors import Determiner
from censor_engine.typing import Image

type Model = Any


class ImageGenreDeterminer(Determiner):
    """
    I made this to help with cases where people might want more precise control
    over when to apply affects.

    For example when I was trying out the edge detection effects, there's a
    stark difference between (AI) hentai and real life porn, primarily due to
    hentai tending to already being outlined if not having some form of cell
    shading which defines the edge. Therefore for example I would use an edge
    detection method with different settings for hentai than I would for porn.
    (probably more aggressive since there's less noise in hentai)

    The model_classifiers attribute lists the original classifiers while the
    broad_groups reduces this down to drawings and real porn, since AI can be
    finicky and debatably (some sick degens think handholding isn't sexual,
    disgusting!), so I reduced "drawings" and "hentai" to just the latter, and
    "porn", "sexy", and "neutral" to "porn".

    Of course this assumes you're only processing porn but this isn't exactly
    a package/tool you use to deny people from seeing your book collection.

    """

    model_name: str = "nsfw_detector"
    model_classifiers: tuple[str, ...] = (
        "drawings",
        "hentai",
        "neutral",
        "porn",
        "sexy",
    )

    broad_groups: dict[str, str] = {  # noqa: RUF012 # TODO: Fix when used
        "hentai": "hentai",
        "real": "porn",
    }

    model_used: Model = None

    def __init__(self):
        pass

    def reduce_results_to_broad_groups(self, results: dict[str, Any]) -> str:
        """
        This method just sums the underling classifiers into broad groups, the
        total score determines the result. This is used to avoid where say a
        hentai drawing is shown but the AI model doesn't recognise tiddies so
        thinks it's just a good christian drawing.

        Neutral is in the "real" category because it's basically the "safe"
        option of IRL porn like "drawing" is, I assume.

        :param dict[str, Any] results: Results from nsfw_detector.predict

        :return str: One of the values from self.broad_groups
        """
        # Assumes All Classifiers are Mentioned, Even 0% Ones
        hentai = results["drawings"] + results["hentai"]
        real = results["neutral"] + results["porn"] + results["sexy"]

        if real >= hentai:
            return self.broad_groups["real"]
        return self.broad_groups["hentai"]

    def determine_image(self, file_image: Image) -> str:  # type: ignore #FIXME: Implement
        """
        Determines the type of pornographic image in question.

        :param str file_path: File location, handled by CensorEngine

        :return str: Returns one of the broad groups; hentai or porn
        """
        return f"{self.model_name}: not working"
        # return self.reduce_results_to_broad_groups(
        #     self.model.classify(
        #         self.model,
        #         file_path,
        #     )
        # )
