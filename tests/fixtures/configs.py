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


# For Testing Processors
CONFIG_DICTS = {
    "baseline": {},
    "body_segmentation": {
        "engine_settings": {
            "extra_ai_model_features": {"body_segmentation": True}
        }
    },
    "focused_roi": {
        "engine_settings": {
            "extra_ai_model_features": {
                "focused_roi": True,
                "body_segmentation": False,
            }
        }
    },
    "clothing_segmentation": {
        "engine_settings": {
            "extra_ai_model_features": {
                "clothes_segmentation": True,
            }
        }
    },
    "depth": {
        "engine_settings": {
            "extra_ai_model_features": {
                "censor_layers": 4,
            }
        }
    },
    "skin_segmentation": {
        "engine_settings": {
            "extra_ai_model_features": {
                "skin_segmentation": True,
            }
        }
    },
}

PROCESSOR_DICT = {
    "censor_test": {
        "detection": {
            "enabled_parts": [
                "YOLOSEG_PERSON_MASK",
                "NUDENET_FACE_FEMALE",
                "NUDENET_ANUS_EXPOSED",
            ],
            "default_settings": {
                "censors": [
                    {"Pixelate": {"factor": 24}},
                    "Outline",
                ]
            },
            "part_settings": {
                "NUDENET_FACE_FEMALE": {
                    "state": "revealed",
                    "mask": "Ellipse",
                },
                "NUDENET_ANUS_EXPOSED": {
                    "state": "revealed",
                    "mask": "Ellipse",
                },
            },
        },
        "image": {"reverse_censor": ["Glow"]},
    },
    "clothing_test": {
        "engine_settings": {
            "extra_ai_model_features": {"skin_segmentation": True}
        },
        "detection": {
            "enabled_parts": [
                "INFERRED_SKIN_MASK",
                "NUDENET_FACE_FEMALE",
                "NUDENET_ANUS_EXPOSED",
            ],
            "default_settings": {
                "censors": [
                    {"Pixelate": {"factor": 24}},
                    {"Outline": {"colour": "HOT_PINK", "thickness": 3}},
                ]
            },
            "part_settings": {
                "NUDENET_FACE_FEMALE": {
                    "state": "revealed",
                    "mask": "Ellipse",
                },
                "NUDENET_ANUS_EXPOSED": {
                    "state": "revealed",
                    "mask": "Ellipse",
                },
            },
        },
        "image": {"reverse_censor": ["Glow"]},
    },
    "bar_amd_depth_test": {
        "engine_settings": {"extra_ai_model_features": {"depth": 0}},
        "detection": {
            "enabled_parts": [
                # "YOLOSEG_PERSON_MASK",
                "NUDENET_FEMALE_BREAST_EXPOSED",
                "NUDENET_FEMALE_GENITALIA_EXPOSED",
                "NUDENET_FACE_FEMALE",
                # "NUDENET_FEET_EXPOSED",
                "NUDENET_BUTTOCKS_EXPOSED",
                "NUDENET_ANUS_EXPOSED",
            ],
            "default_settings": {
                "censors": [
                    {"Pixelate": {"factor": 24}},
                    {"Outline": {"colour": "black"}},
                ],
                "mask": "Bar",
            },
            "part_settings": {
                "NUDENET_ANUS_EXPOSED": {
                    "state": "revealed",
                    "mask": "Ellipse",
                }
            },
        },
    },
}
