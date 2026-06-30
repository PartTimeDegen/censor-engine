import pytest

from censor_engine.models.enums import MergeMethod
from censor_engine.models.models_library.configs.settings._image import (
    ImageSettings,
    _Merging,
)
from censor_engine.structs.censors import Censor


class TestMerging:
    def test_initiate(self):
        m = _Merging()
        assert m.method == MergeMethod.NONE
        assert m.merge_range == 0.0

    class TestFields:
        class TestMethod:
            def test_baseline(self):
                for method in MergeMethod:
                    assert _Merging(method=method.name).method == method  # type: ignore

            def test_missing_method(self):
                with pytest.raises(AttributeError):
                    _Merging(method="something else")  # type: ignore


class TestImageSettings:
    def test_initiate(self):
        i = ImageSettings()

        assert i.merging == _Merging()
        assert i.reverse_censor == []

    class TestFields:
        class TestReverseCensor:
            def test_baseline(self):
                i = ImageSettings(reverse_censor="blur")  # type: ignore

                assert i.reverse_censor == [Censor("blur")]
