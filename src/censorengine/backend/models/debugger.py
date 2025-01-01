from dataclasses import dataclass, field
from enum import IntEnum
from functools import wraps
import os
import timeit
import onnxruntime as ort  # type: ignore

import cv2


class DebugLevels(IntEnum):
    NONE = 0
    BASIC = 1
    DETAILED = 2
    ADVANCED = 3
    FULL = 4


@dataclass
class TimerSchema:
    name: str
    timestamp: float
    duration: float

    def __str__(self):
        return f"{self.name} = {self.duration:0.2d} seconds"

    def __repr__(self):
        return f"{self.name} = {self.duration:0.2d} seconds ({self.timestamp})"


class Debugger:
    # THIS IS USED, NOT INHERITED
    # General
    debug_name: str = "MISSING_NAME"
    debug_level: DebugLevels = DebugLevels.NONE

    # Masks

    # Timer
    time_logger: list[TimerSchema] = []

    # # Stats
    stats_duration: float = field(init=False)

    def __init__(self, name: str, level: DebugLevels):
        self.debug_name = name.upper()
        self.debug_level = level
        self.time_logger = []

    # @staticmethod # FIXME: use Python's module
    # def time_function(func):
    #     @wraps(func)
    #     def wrapper(*args, **kwargs):
    #         # Function Proper
    #         before = timeit.default_timer()
    #         func(*args, **kwargs)
    #         after = timeit.default_timer()
    #         duration = after - before

    #         # Time Management
    #         timer = TimerSchema(
    #             name=func.__name__,
    #             timestamp=before,
    #             duration=duration,
    #         )
    #         if not Debugger.time_logger:
    #             Debugger.time_logger = [timer]
    #         else:
    #             Debugger.time_logger += [timer]

    #     return wrapper

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

    # Display Information
    def display_onnx_info(self):
        if self.debug_level >= DebugLevels.DETAILED:
            print(f"[ DEBUG {self.debug_name}: ONNX_INFO")
            print(f"[ - Onnxruntime device: {ort.get_device()}")
            print(
                f"[ - Ort available providers: {ort.get_available_providers()}"
            )
            print()

    def display_times(self):
        if self.debug_level >= DebugLevels.BASIC:
            print(f"[ DEBUG {self.debug_name}: FUNCTION TIMES")

            for time in reversed(
                sorted(self.debug_time_logger, key=lambda x: x.duration)
            ):
                print(f"[ - {time}")
            print()


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
        additional_text = (
            f", {relative:2.1f}x minimum" if min_duration > 0.001 else ""
        )
        print(
            f"{time_id:>2} {time_name}{gap}: {duration:0.3f} seconds{additional_text}"
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
