import pytest

from censor_engine.core.tools.debugger import DebugLevel
from censor_engine.models.models_library.configs.settings._development import (
    DevelopmentSettings,
)


class TestMerging:
    def test_initiate(self):
        ds = DevelopmentSettings()
        assert ds.debug_level == DebugLevel.NONE

    class TestFields:
        class TestMethod:
            def test_baseline(self):
                for level in DebugLevel:
                    assert (
                        DevelopmentSettings(debug_level=level.name).debug_level
                        == level
                    )  # type: ignore

            def test_missing_method(self):
                with pytest.raises(AttributeError):
                    DevelopmentSettings(debug_level="something else")  # type: ignore
