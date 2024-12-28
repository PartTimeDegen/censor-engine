from dataclasses import dataclass, field
import os
import timeit

import cv2


class _TimeLogger: ...


@dataclass
class Debugger:
    # General
    debug_level: int = 0

    # Masks

    # Timer
    debug_time_logger: list[tuple[int, str, float, float]] = field(
        default_factory=list[tuple[int, str, float, float]]
    )

    # # Stats
    stats_duration: float = field(init=False)

    def save_masks(self, label=None):
        def arg_layer():
            def wrapper():
                # Check for Existing Variables
                def check_attr(attr):
                    try:
                        getattr(self, attr)
                        return True
                    except AttributeError:
                        return False

                # Handle Counter
                if check_attr("_debug_mask_counter"):
                    self._debug_mask_counter += 1
                else:
                    self._debug_mask_counter = 0

                # Handle Folder Name
                folder_list = [".debug", label, self.file_image_name]
                if label:
                    folder_list[1] = (
                        f"{self._debug_mask_counter}_{folder_list[1]}"
                    )

                for part in self.parts:
                    # Get Names
                    folder = os.path.join(
                        *[fold for fold in folder_list if fold]
                    )
                    file = os.path.join(
                        folder,
                        f"{part.part_id}_{part.part_name}_{part.state}.jpg",
                    )

                    # Make Folder
                    os.makedirs(folder, exist_ok=True)

                    # Write File
                    cv2.imwrite(file, part.mask)

                return wrapper

            return arg_layer

    def log_time(self): ...


def _debug_save_time(self, name):
    new_id = self.debug_time_logger[-1][0] + 1
    self.debug_time_logger.append(
        (
            new_id,
            name,
            timeit.default_timer(),
            timeit.default_timer() - self.debug_time_logger[-1][2],
        )
    )


def _display_debug_times(self):
    durations = [time[-1] for time in self.debug_time_logger]
    min_duration = min([time for time in durations if time > 0.001])
    if min_duration > 0.001:
        times = [
            list(time) + [time[-1] / min_duration]
            for time in self.debug_time_logger
        ]
    else:
        times = [list(time) + [""] for time in self.debug_time_logger]

    times = reversed(sorted(times, key=lambda x: x[3]))

    print("")
    print("=== Debug Durations ===")
    for time_id, time_name, _, duration, relative in times:
        if time_name == "init":
            continue

        spacing = 40
        gap = " " * (spacing - len(time_name))
        addtional_text = (
            f", {relative:2.1f}x minimum" if min_duration > 0.001 else ""
        )
        print(
            f"{time_id:>2} {time_name}{gap}: {duration:0.3f} seconds{addtional_text}"
        )


def _debug_save_masks(self, label=None):
    def check_attr(attr):
        try:
            getattr(self, attr)
            return True
        except AttributeError:
            return False

    # Handle Counter
    if check_attr("_debug_mask_counter"):
        self._debug_mask_counter += 1
    else:
        self._debug_mask_counter = 0

    folder_list = [
        ".debug",
        label,
        self.file_image_name,
    ]
    if label:
        folder_list[1] = f"{self._debug_mask_counter}_{folder_list[1]}"

    for part in self.parts:
        # Get Names
        folder = os.path.join(*[fold for fold in folder_list if fold])
        file = os.path.join(
            folder,
            f"{part.part_id}_{part.part_name}_{part.state}.jpg",
        )

        # Make Folder
        os.makedirs(
            folder,
            exist_ok=True,
        )

        # Write File
        cv2.imwrite(file, part.mask)
