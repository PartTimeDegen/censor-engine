import statistics
from censorengine.libs.detector_library.catalogue import (
    enabled_detectors,
    enabled_determiners,
)
from censorengine.libs.shape_library.catalogue import shape_catalogue
from censorengine.libs.style_library.catalogue import style_catalogue


class ComponentReporting:
    # Reporting
    def get_detectors(self) -> list[str]:
        return [detector.model_name for detector in enabled_detectors]

    def get_determiners(self) -> list[str]:
        return [detector.model_name for detector in enabled_determiners]

    def get_shapes(self) -> list[str]:
        return list(shape_catalogue.keys())

    def get_censor_styles(self) -> list[str]:
        return list(style_catalogue.keys())

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

        max_key_length = max(len(key) for key in dict_stats) + 4
        print()
        print("Run Statistics:")
        for key, value in dict_stats.items():
            if key != "CoV":
                print(f"- {key:<{max_key_length}}: {value * 1000:>6.3f} ms")
            else:
                print(f"- {key:<{max_key_length}}: {value:>2.3%}")
