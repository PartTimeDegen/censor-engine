from pathlib import Path

from censor_engine.models.libraries.configs.settings._file_handling import (
    FileHandingSettings,
)


def test_str_to_path():
    fh = FileHandingSettings.model_validate(
        {"folders": {"uncensored": "something"}}
    )  # type: ignore

    assert fh.folders.uncensored == Path("something")
