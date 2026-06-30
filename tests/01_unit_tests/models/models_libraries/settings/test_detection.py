import pytest
from pydantic import ValidationError

from censor_engine.models.enums import PartState
from censor_engine.models.models_library.configs._constants import DEFAULT_MASK
from censor_engine.models.models_library.configs.settings._detection import (
    DetectionSettings,
    _PartSettings,
)


class TestPartSettings:
    def test_initiate(self):
        _PartSettings()

    class TestFields:
        class TestMask:
            def test_baseline(self):
                assert _PartSettings().mask == DEFAULT_MASK

            def test_custom(self):
                assert _PartSettings(protection_mask="Box").mask == "Box"

        class TestMinimumScore:
            def test_baseline(self):
                assert _PartSettings().minimum_score == 0.0

            def test_custom(self):
                assert _PartSettings(minimum_score=0.5).minimum_score == 0.5

            def test_over_limit(self):
                with pytest.raises(ValidationError):
                    _PartSettings(minimum_score=1.1)

            def test_under_limit(self):
                with pytest.raises(ValidationError):
                    _PartSettings(minimum_score=-0.1)

        class TestState:
            def test_baseline(self):
                assert _PartSettings().state == PartState.UNPROTECTED

            def test_custom(self):
                for part in PartState:
                    assert _PartSettings(state=str(part.name)).state == part  # type: ignore

            def test_missing_state(self):
                with pytest.raises(AttributeError):
                    ps = _PartSettings(state="doesn't exist").state  # type: ignore

        class TestProtectionMask:
            def test_baseline(self):
                assert _PartSettings().protection_mask is None

            def test_custom(self):
                assert (
                    _PartSettings(protection_mask="Box").protection_mask
                    == "Box"
                )

        class TestFade:
            def test_baseline(self):
                assert _PartSettings().fade == 0.0

            def test_custom(self):
                assert _PartSettings(fade=0.5).fade == 0.5

            def test_over_limit(self):
                with pytest.raises(ValidationError):
                    _PartSettings(fade=1.1)

            def test_under_limit(self):
                with pytest.raises(ValidationError):
                    _PartSettings(fade=-0.1)

        class TestUseGlobalArea:
            def test_baseline(self):
                assert _PartSettings().use_global_area == True

            def test_false(self):
                assert (
                    _PartSettings(use_global_area=False).use_global_area
                    == False
                )


class TestDetectionSettings:
    def test_initiate(self):
        DetectionSettings()

    class TestEnabledParts:
        def test_baseline(self):
            ds = DetectionSettings(enabled_parts=["FEMALE_FACE"])
            assert ds.enabled_parts == ["FEMALE_FACE"]

        # TODO: This needs to be implemented, need the register updated
        # def test_all_word(self):
        # ds = DetectionSettings(enabled_parts=["FEMALE_FACE"])
        # def test_part_accepted(self): ...
        # ds = DetectionSettings(enabled_parts=["PLACEHOLDER"])

    class TestDefaultSettings:
        def test_baseline(self):
            ds = DetectionSettings(
                default_settings={
                    "mask": "Circle",
                    "state": "protected",
                    "fade": 0.5,
                    "protection_mask": "placeholder",
                }  # type: ignore
            )
            assert ds.default_settings.mask == "Circle"
            assert ds.default_settings.state == PartState.PROTECTED
            assert ds.default_settings.fade == 0.5
            assert ds.default_settings.protection_mask == "placeholder"

    class TestParts:
        def test_baseline(self):
            ds = DetectionSettings(
                parts={
                    "Thing": {
                        "mask": "Circle",
                        "state": "protected",
                        "fade": 0.5,
                        "protection_mask": "placeholder",
                    }
                }  # type: ignore
            )
            assert len(ds.parts) == 1
            assert ds.parts["Thing"].mask == "Circle"
            assert ds.parts["Thing"].state == PartState.PROTECTED
            assert ds.parts["Thing"].fade == 0.5
            assert ds.parts["Thing"].protection_mask == "placeholder"

        def test_defaults(self):
            ds = DetectionSettings(
                parts={"Thing": {}}  # type: ignore
            )
            assert len(ds.parts) == 1
            assert ds.parts["Thing"].mask == DEFAULT_MASK
            assert ds.parts["Thing"].state == PartState.UNPROTECTED
            assert ds.parts["Thing"].fade == 0.0
            assert ds.parts["Thing"].protection_mask is None

        def test_empty(self):
            ds = DetectionSettings(parts={})  # type: ignore
            assert len(ds.parts) == 0

        def test_multiple_parts(self):
            ds = DetectionSettings(
                parts={
                    "Thing": {
                        "mask": "Circle",
                        "state": "protected",
                        "fade": 0.5,
                        "protection_mask": "placeholder",
                    },
                    "Thing2": {},
                }  # type: ignore
            )
            assert len(ds.parts) == 2
            assert ds.parts["Thing"].mask == "Circle"
            assert ds.parts["Thing"].state == PartState.PROTECTED
            assert ds.parts["Thing"].fade == 0.5
            assert ds.parts["Thing"].protection_mask == "placeholder"
            assert ds.parts["Thing2"].mask == DEFAULT_MASK
            assert ds.parts["Thing2"].state == PartState.UNPROTECTED
            assert ds.parts["Thing2"].fade == 0.0
            assert ds.parts["Thing2"].protection_mask is None
