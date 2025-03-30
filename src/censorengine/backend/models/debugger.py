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
    time_id: int = itertools.count(start=1)  # type: ignore

    def __str__(self):
        time = f"{self.duration*1_000:> 10.6f}"

        # Decimal Sep
        # # Splitting into Components
        whole, decimal = time.split(".")

        # # Adding Missing Zeros

        if (missing_zeros := len(decimal) % 3) != 0:
            decimal += "0" * missing_zeros

        # # Adding Spacers
        offset = False
        if decimal[-1] == "0":
            offset = True
            decimal = list(decimal)
            decimal[-1] = "1"
            decimal = "".join(decimal)
        fixed_decimal = f"{int(decimal[::-1]):,}".replace(",", " ")[::-1]
        if offset:
            fixed_decimal = list(fixed_decimal)
            fixed_decimal[-1] = "0"
            fixed_decimal = "".join(fixed_decimal)

        # # Merging
        fixed_time = f"{whole}.{fixed_decimal}"
        return f"({fixed_time}ms) {self.name}"

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
            logged.duration for logged in self.time_logger if logged.duration > min_time
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
