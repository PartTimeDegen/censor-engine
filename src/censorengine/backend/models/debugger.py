from dataclasses import dataclass, field
from enum import IntEnum
import itertools
import os
import time
from typing import Iterable, Optional
import onnxruntime as ort  # type: ignore

import cv2


class DebugLevels(IntEnum):
    """
    This function determines the levels of debug that are active (stacks):
        -   NONE:
            -   Nothing ever happens (I'm all in)
        -   BASIC:
            -   Found Parts
            -   Censor Time
        -   DETAILED:
            -   GPU and ONNX Info
            -   Times of the Functions
            -   Censor Part Info
            -   Config Info
        -   ADVANCED:
            -   Shape IDs
            -   Merge IDs
            -   Used Censors and Styles
        -   FULL:
            -   Mask Breakdowns

    """

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

    program_start_time: bool = field(default=False)
    time_id: Iterable[int] = itertools.count(start=1)

    def __str__(self):
        return f"({self.duration: >8.3f}s) {self.name}"

    def __repr__(self):
        return f"{self.name} ({self.timestamp})"

    def __post_init__(self):
        if self.program_start_time:
            self.time_id = 1
        else:
            self.time_id = next(TimerSchema.time_id)  # type: ignore


class Debugger:
    # THIS IS USED, NOT INHERITED
    # General
    debug_name: str = "MISSING_NAME"
    debug_level: DebugLevels = DebugLevels.NONE

    # Masks

    # Timer
    time_logger: list[TimerSchema] = []
    temp_time_holder: Optional[tuple[str, float]] = None  # Name, Time (Start)
    program_start: TimerSchema

    # # Stats
    stats_duration: float = field(init=False)

    def __init__(self, name: str, level: DebugLevels):
        self.debug_name = name.upper()
        self.debug_level = level
        self.time_logger = []
        self.temp_time_holder = None

    def time_total_start(self):
        self.program_start = TimerSchema(
            name=self.debug_name,
            timestamp=time.time(),
            duration=0.0,
            program_start_time=True,
        )

    def time_total_end(self):
        self.program_start.duration = time.time() - self.program_start.timestamp

        self.time_logger.append(self.program_start)

    def time_start(self, name: str):
        if self.debug_level < DebugLevels.DETAILED:
            return

        if self.temp_time_holder:
            raise TypeError("Missing Stop for Timer")

        self.temp_time_holder = (name, time.time())

    def time_stop(self):
        if self.debug_level < DebugLevels.DETAILED:
            return

        if not self.temp_time_holder:
            raise TypeError("Missing Start for Timer")

        duration = time.time() - self.temp_time_holder[1]

        self.time_logger.append(
            TimerSchema(
                name=self.temp_time_holder[0],
                timestamp=self.temp_time_holder[1],
                duration=duration,
            )
        )

        self.temp_time_holder = None

    @DeprecationWarning
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
                    folder_list[1] = f"{self._debug_mask_counter}_{folder_list[1]}"

                for part in self.parts:
                    # Get Names
                    folder = os.path.join(*[fold for fold in folder_list if fold])
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
        """
        This function simply displays the AI information such as the device and
        provider. Used to tell if GPU-acceleration is used.

        Level: DETAILED

        """
        if self.debug_level >= DebugLevels.DETAILED:
            print(f"[ DEBUG {self.debug_name}: ONNX_INFO")
            print(f"[ - Onnxruntime device: {ort.get_device()}")
            print(f"[ - Ort available providers: {ort.get_available_providers()}")
            print()

    def display_times(self):
        if self.debug_level < DebugLevels.BASIC:
            return

        print(f"[ DEBUG {self.debug_name}: FUNCTION TIMES")

        sorted_times = list(
            reversed(sorted(self.time_logger, key=lambda x: x.duration))
        )
        if len(sorted_times) == 1:
            print(sorted_times[0])
            return

        max_time = max(logged.duration for logged in self.time_logger)
        min_time = min(logged.duration for logged in self.time_logger)
        min_time = min(
            logged.duration
            for logged in self.time_logger
            if logged.duration > (min_time * 2)
        )

        for proc_time in sorted_times:
            if proc_time.program_start_time:
                print(f"{proc_time}")
            else:
                factor = int(proc_time.duration / min_time)
                print(
                    f"[ {proc_time.time_id:02d}) {proc_time} [{factor if factor > 0 else 1}x / {proc_time.duration/max_time:2.1%}]"
                )
        print()


# def _debug_save_time(self, name):
#     new_id = self.debug_time_logger[-1][0] + 1
#     self.debug_time_logger.append(
#         (
#             new_id,
#             name,
#             timeit.default_timer(),
#             timeit.default_timer() - self.debug_time_logger[-1][2],
#         )
#     )


# def _display_debug_times(self):
#     durations = [time[-1] for time in self.debug_time_logger]
#     min_duration = min([time for time in durations if time > 0.001])
#     if min_duration > 0.001:
#         times = [
#             list(time) + [time[-1] / min_duration]
#             for time in self.debug_time_logger
#         ]
#     else:
#         times = [list(time) + [""] for time in self.debug_time_logger]

#     times = reversed(sorted(times, key=lambda x: x[3]))

#     print("")
#     print("=== Debug Durations ===")
#     for time_id, time_name, _, duration, relative in times:
#         if time_name == "init":
#             continue

#         spacing = 40
#         gap = " " * (spacing - len(time_name))
#         additional_text = (
#             f", {relative:2.1f}x minimum" if min_duration > 0.001 else ""
#         )
#         print(
#             f"{time_id:>2} {time_name}{gap}: {duration:0.3f} seconds{additional_text}"
#         )


# def _debug_save_masks(self, label=None):
#     def check_attr(attr):
#         try:
#             getattr(self, attr)
#             return True
#         except AttributeError:
#             return False

#     # Handle Counter
#     if check_attr("_debug_mask_counter"):
#         self._debug_mask_counter += 1
#     else:
#         self._debug_mask_counter = 0

#     folder_list = [
#         ".debug",
#         label,
#         self.file_image_name,
#     ]
#     if label:
#         folder_list[1] = f"{self._debug_mask_counter}_{folder_list[1]}"

#     for part in self.parts:
#         # Get Names
#         folder = os.path.join(*[fold for fold in folder_list if fold])
#         file = os.path.join(
#             folder,
#             f"{part.part_id}_{part.part_name}_{part.state}.jpg",
#         )

#         # Make Folder
#         os.makedirs(
#             folder,
#             exist_ok=True,
#         )

#         # Write File
#         cv2.imwrite(file, part.mask)


class TempTimer:
    def __init__(self, name):
        self.debugger = Debugger(name, DebugLevels.DETAILED)
        self.debugger.time_total_start()

    def time_start(self, name):
        self.debugger.time_start(name)

    def time_stop(self):
        self.debugger.time_stop()

    def time_end(self):
        self.debugger.time_total_end()
        self.debugger.display_times()
