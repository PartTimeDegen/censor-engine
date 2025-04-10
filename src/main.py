import pathlib
from censorengine import CensorEngine

if __name__ == "__main__":
    file_path = pathlib.Path(__file__).parent.resolve()

    censor_engine = CensorEngine(str(file_path))

    censor_engine.start()
    print()
