import pytest

from censor_engine.models.core.tools.debugger.enums import DebugLevel
from censor_engine.models.enums import MergeMethod, PartState
from censor_engine.models.libraries.configs._validators import (
    convert_debug_level,
    convert_merge_method,
    convert_state,
    normalise_censors,
    normalise_margins,
)
from censor_engine.structs.censors import Censor


class TestNormalisers:
    class TestNormaliseCensors:
        def test_string(self):
            input_data = "thing"
            expected_output = [Censor(input_data)]
            assert normalise_censors(input_data) == expected_output

        def test_dictionary(self):
            input_data = {"effect": "thing", "parameters": {}}
            expected_output = [Censor(**input_data)]
            assert normalise_censors(input_data) == expected_output

        def test_list_of_strings(self):
            input_data = ["t", "h", "i", "n", "g"]
            expected_output = [Censor(data) for data in input_data]
            assert normalise_censors(input_data) == expected_output

        def test_list_of_dictionaries(self):
            input_data = [
                {"effect": "t", "parameters": {}},
                {"effect": "h", "parameters": {}},
                {"effect": "i", "parameters": {}},
                {"effect": "n", "parameters": {}},
                {"effect": "g", "parameters": {}},
            ]
            expected_output = [Censor(**data) for data in input_data]
            assert normalise_censors(input_data) == expected_output

    class TestNormaliseMargins:
        def test_float(self):
            input_data = 1.0
            assert normalise_margins(input_data) == {
                "height": 1.0,
                "width": 1.0,
            }

        def test_int(self):
            input_data = 1
            assert normalise_margins(input_data) == {
                "height": 1.0,
                "width": 1.0,
            }

        def test_half_dictionary(self):
            input_data = {"height": 1.0}
            assert normalise_margins(input_data) == {
                "height": 1.0,
            }

        def test_full_dictionary(self):
            input_data = {"height": 1.0, "width": 1.0}
            assert normalise_margins(input_data) == {
                "height": 1.0,
                "width": 1.0,
            }


class TestConvertors:
    class TestMergeMethod:
        def test_baseline(self):
            for state in MergeMethod:
                name = state.name
                assert state == convert_merge_method(name)

        def test_lowercase(self):
            for state in MergeMethod:
                name = state.name.lower()
                assert state == convert_merge_method(name)

        def test_wrong_type(self):
            with pytest.raises(TypeError):
                convert_merge_method(3)  # type: ignore

        def test_wrong_word(self):
            with pytest.raises(AttributeError):
                convert_merge_method("SOMETHING ELSE")

    class TestDebugLevel:
        def test_baseline(self):
            for state in DebugLevel:
                name = state.name
                assert state == convert_debug_level(name)

        def test_lowercase(self):
            for state in DebugLevel:
                name = state.name.lower()
                assert state == convert_debug_level(name)

        def test_wrong_type(self):
            with pytest.raises(TypeError):
                convert_debug_level(3)  # type: ignore

        def test_wrong_word(self):
            with pytest.raises(AttributeError):
                convert_debug_level("SOMETHING ELSE")

    class TestPartState:
        def test_baseline(self):
            for state in PartState:
                name = state.name
                assert state == convert_state(name)

        def test_lowercase(self):
            for state in PartState:
                name = state.name.lower()
                assert state == convert_state(name)

        def test_wrong_type(self):
            with pytest.raises(TypeError):
                convert_state(3)  # type: ignore

        def test_wrong_word(self):
            with pytest.raises(AttributeError):
                convert_state("SOMETHING ELSE")
