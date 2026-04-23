import statistics

from censor_engine.libs.detectors import (
    enabled_detectors,
    enabled_determiners,
)
from censor_engine.libs.registries import EffectRegistry, MaskRegistry
from censor_engine.models.structs import Mixin


class MixinReporting(Mixin):
    # Reporting
    def get_detectors(self) -> list[str]:
        return [detector.model_name for detector in enabled_detectors]

    def get_determiners(self) -> list[str]:
        return [detector.model_name for detector in enabled_determiners]

    def get_masks(self) -> list[str]:
        return list(MaskRegistry.get_all().keys())

    def get_censor_effects(self) -> list[str]:
        return list(EffectRegistry.get_all().keys())

    def display_bulk_stats(self, durations: list[float]) -> None:
        mean = statistics.mean(durations)

        dict_stats = {
            "Mean": mean,
            "Median": statistics.median(durations),
            "Min": min(durations),
            "Max": max(durations),
            "Range": max(durations) - min(durations),
        }
        if len(durations) > 1:
            stdev = statistics.stdev(durations)
            coefficient_of_variation = stdev / mean
            dict_stats["Stdev"] = stdev
            dict_stats["CoV"] = coefficient_of_variation

        max(len(key) for key in dict_stats) + 4
        for key in dict_stats:
            if key != "CoV":
                pass
            else:
                pass
