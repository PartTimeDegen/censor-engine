import pathlib
from censorengine import CensorEngine

if __name__ == "__main__":
    file_path = pathlib.Path(__file__).parent.resolve()

    CensorEngine(
        file_path,
        show_duration=True,
        debug_log_time=True,
    )
