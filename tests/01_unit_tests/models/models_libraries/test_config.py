# tests/test_config.py

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import __main__
from censor_engine.models.models_core.tools.debugger.enums import DebugLevels
from censor_engine.models.models_library.configs.config import Config


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def main_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a fake project containing main.py."""
    project = tmp_path / "project"
    project.mkdir()

    main_file = project / "main.py"
    main_file.touch()

    monkeypatch.setattr(__main__, "__file__", str(main_file))

    return project


class TestConfig:
    class TestClassMethods:
        class TestFromDict:
            def test_baseline(self):
                config = Config.from_dict({})
                assert isinstance(config, Config)

            def test_values(self):
                config = Config.from_dict(
                    {
                        "version": 123,
                        "development": {"debug_level": "DETAILED"},
                    }
                )
                assert config.version == 123
                assert config.development.debug_level == DebugLevels.DETAILED

        class TestFromYAML:
            def test_builtin_config(self):
                config = Config.from_yaml("basic/default.yml")
                assert config.version == 2

            def test_custom_config(self, main_dir: Path):
                config_path = main_dir / "config.yml"
                write_yaml(config_path, {"version": 99})

                config = Config.from_yaml("config.yml")
                assert config.version == 99

            def test_duplicates_prefer_builtin(self, main_dir: Path):
                config_path = main_dir / "basic/default.yml"
                config_path.parent.mkdir(exist_ok=True)
                write_yaml(config_path, {"version": 99})

                config = Config.from_yaml("basic/default.yml")

                assert config.version == 2

            def test_missing_yaml(self, main_dir: Path):
                with pytest.raises(FileNotFoundError):
                    Config.from_yaml("missing.yaml")
