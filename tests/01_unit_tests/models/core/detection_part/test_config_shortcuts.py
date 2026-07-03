from copy import deepcopy

import pytest

from censor_engine.models.core.detection_part._config_shortcuts import (
    ConfigShortcuts,
)
from censor_engine.models.enums import MergeMethod
from censor_engine.models.libraries.configs.config import (
    Config,
)


@pytest.fixture
def config():
    return Config.from_dict(
        {
            "detection": {
                "enabled_parts": [
                    "PLACEHOLDER_ONE",
                    "PLACEHOLDER_TWO",
                    "PLACEHOLDER_THREE",
                ],
                "parts": {
                    "PLACEHOLDER_ONE": {},
                    "PLACEHOLDER_TWO": {},
                    "PLACEHOLDER_THREE": {},
                },
            }
        }
    )


@pytest.fixture
def config_populated():
    return Config.from_dict(
        {
            "detection": {
                "enabled_parts": [
                    "PLACEHOLDER_ONE",
                    "PLACEHOLDER_TWO",
                    "PLACEHOLDER_THREE",
                    "PLACEHOLDER_FOUR",
                ],
                "default_settings": {
                    "minimum_score": 0.75,
                    "margins": 0.2,
                    "censors": ["blur"],
                    "mask": "Box",
                    "protection_mask": "Circle",
                },
                "parts": {
                    "PLACEHOLDER_ONE": {},
                    "PLACEHOLDER_TWO": {},
                    "PLACEHOLDER_THREE": {},
                    "PLACEHOLDER_FOUR": {},
                },
            },
            "groups": {
                "merging": [
                    ["PLACEHOLDER_ONE", "PLACEHOLDER_TWO"],
                    ["PLACEHOLDER_THREE", "PLACEHOLDER_FOUR"],
                ],
                "persistance": [
                    ["PLACEHOLDER_ONE", "PLACEHOLDER_TWO"],
                ],
            },
        }
    )


@pytest.fixture
def config_merge_all(config_populated: Config) -> Config:
    new_config = deepcopy(config_populated)
    new_config.image.merging.method = MergeMethod.ALL
    return new_config


@pytest.fixture
def shortcuts(config_populated: Config) -> ConfigShortcuts:
    return ConfigShortcuts(
        detection_settings=config_populated.detection,
        part_settings=config_populated.detection.parts["PLACEHOLDER_ONE"],
        groups=config_populated.groups,
        image_settings=config_populated.image,
    )


@pytest.fixture
def shortcuts_merge_all(config_merge_all: Config) -> ConfigShortcuts:
    return ConfigShortcuts(
        detection_settings=config_merge_all.detection,
        part_settings=config_merge_all.detection.parts["PLACEHOLDER_ONE"],
        groups=config_merge_all.groups,
        image_settings=config_merge_all.image,
    )


class TestConfigShortcuts:
    def test_initiate(self, config: Config):
        ConfigShortcuts(
            detection_settings=config.detection,
            part_settings=config.detection.parts["PLACEHOLDER_ONE"],
            groups=config.groups,
            image_settings=config.image,
        )

    class TestGeneratedFields:
        class TestMinimumScore:
            def test_baseline(self, shortcuts: ConfigShortcuts):
                assert shortcuts.minimum_score == 0.75

        class TestMargins:
            def test_baseline(self, shortcuts: ConfigShortcuts):
                assert shortcuts.margins.height == 0.2
                assert shortcuts.margins.width == 0.2

        class TestMaskName:
            def test_baseline(self, shortcuts: ConfigShortcuts):
                assert shortcuts.mask_name == "Box"

        class TestProtectionMaskName:
            def test_baseline(self, shortcuts: ConfigShortcuts):
                assert shortcuts.protection_mask_name == "Circle"

        class TestGroupsMerge:
            def test_baseline(self, shortcuts: ConfigShortcuts):
                assert shortcuts.groups_merge == [
                    ["PLACEHOLDER_ONE", "PLACEHOLDER_TWO"],
                    ["PLACEHOLDER_THREE", "PLACEHOLDER_FOUR"],
                ]

            def test_merge_all(
                self,
                shortcuts_merge_all: ConfigShortcuts,
            ):
                assert shortcuts_merge_all.groups_merge == [
                    [
                        "PLACEHOLDER_ONE",
                        "PLACEHOLDER_TWO",
                        "PLACEHOLDER_THREE",
                        "PLACEHOLDER_FOUR",
                    ],
                ]

        class TestGroupsPersistence:
            def test_baseline(self, shortcuts: ConfigShortcuts):
                assert shortcuts.groups_persistance == [
                    ["PLACEHOLDER_ONE", "PLACEHOLDER_TWO"],
                ]

        class TestMergeMethod:
            def test_baseline(self, shortcuts: ConfigShortcuts):
                assert shortcuts.merge_method == MergeMethod.NONE

            def test_merge_all(
                self,
                shortcuts_merge_all: ConfigShortcuts,
            ):
                assert shortcuts_merge_all.merge_method == MergeMethod.ALL

    class TestPostInit:
        class TestMergeMethodAll:
            def test_overrides_merge_groups(
                self,
                shortcuts_merge_all: ConfigShortcuts,
            ):
                assert shortcuts_merge_all.groups_merge == [
                    shortcuts_merge_all.detection_settings.enabled_parts
                ]
