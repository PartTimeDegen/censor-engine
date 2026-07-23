import pytest

from censor_engine.models.libraries.configs.config import Config


@pytest.fixture
def groups() -> list[list[str]]:
    return [
        ["boobs", "bum"],
        ["car", "vehicle"],
    ]


@pytest.fixture
def single_part(groups: list[list[str]]) -> str:
    return groups[0][0]


@pytest.fixture
def config_part_minimum(single_part) -> Config:
    return Config.from_dict({"detection": {"enabled_parts": [single_part]}})


@pytest.fixture
def config_with_parts(groups, single_part) -> Config:
    return Config.from_dict(
        {
            "groups": {"merging": groups, "persistance": groups},
            "detection": {
                "enabled_parts": [single_part],
                "parts": {single_part: {"margins": 0.2}},
            },
        }
    )
