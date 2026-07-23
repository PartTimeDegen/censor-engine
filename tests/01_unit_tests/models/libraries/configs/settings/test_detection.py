import pytest
from pydantic import ValidationError

from censor_engine.models.enums import PartState
from censor_engine.models.libraries.configs._constants import DEFAULT_MASK
from censor_engine.models.libraries.configs.settings._detection import (
    DetectionSettings,
    PartSettings,
)
from censor_engine.models.libraries.configs.settings.schemas import Margins
from censor_engine.structs.censors import Censor


class TestPartSettings:
    def test_initiate(self):
        PartSettings()

    class TestFields:
        class TestMask:
            def test_baseline(self):
                assert PartSettings().mask == DEFAULT_MASK

            def test_custom(self):
                assert PartSettings(protection_mask="Box").mask == "Box"

        class TestMinimumScore:
            def test_baseline(self):
                assert PartSettings().minimum_score == 0.0

            def test_custom(self):
                assert PartSettings(minimum_score=0.5).minimum_score == 0.5

            def test_over_limit(self):
                with pytest.raises(ValidationError):
                    PartSettings(minimum_score=1.1)

            def test_under_limit(self):
                with pytest.raises(ValidationError):
                    PartSettings(minimum_score=-0.1)

        class TestState:
            def test_baseline(self):
                assert PartSettings().state == PartState.UNPROTECTED

            def test_custom(self):
                for part in PartState:
                    assert PartSettings(state=str(part.name)).state == part  # type: ignore

            def test_missing_state(self):
                with pytest.raises(AttributeError):
                    ps = PartSettings(state="doesn't exist").state  # type: ignore

        class TestProtectionMask:
            def test_baseline(self):
                assert PartSettings().protection_mask is None

            def test_custom(self):
                assert (
                    PartSettings(protection_mask="Box").protection_mask
                    == "Box"
                )

        class TestFade:
            def test_baseline(self):
                assert PartSettings().fade == 0.0

            def test_custom(self):
                assert PartSettings(fade=0.5).fade == 0.5

            def test_over_limit(self):
                with pytest.raises(ValidationError):
                    PartSettings(fade=1.1)

            def test_under_limit(self):
                with pytest.raises(ValidationError):
                    PartSettings(fade=-0.1)

        class TestUseGlobalArea:
            def test_baseline(self):
                assert PartSettings().use_global_area == True

            def test_false(self):
                assert (
                    PartSettings(use_global_area=False).use_global_area
                    == False
                )

        class TestCensors:
            def test_baseline(self):
                ps = PartSettings(censors=[{"effect": "blur"}])
                assert ps.censors == [Censor(effect="blur")]

        class TestMargins:
            def test_baseline(self):
                ps = PartSettings()
                assert isinstance(ps.margins, Margins)
                assert ps.margins == Margins()

            def test_float(self):
                ps = PartSettings(margins=0.2)
                assert isinstance(ps.margins, Margins)
                assert ps.margins == Margins(width=0.2, height=0.2)

            def test_int(self):
                ps = PartSettings(margins=1)
                assert isinstance(ps.margins, Margins)
                assert ps.margins == Margins(width=1.0, height=1.0)

            def test_dict(self):
                ps = PartSettings(margins={"width": 0.2, "height": 0.2})
                assert isinstance(ps.margins, Margins)
                assert ps.margins == Margins(width=0.2, height=0.2)

        class TestTrackingMargins:
            def test_baseline(self):
                ps = PartSettings()
                assert isinstance(ps.tracking_margin, Margins)
                assert ps.tracking_margin == Margins()

            def test_float(self):
                ps = PartSettings(tracking_margin=0.2)
                assert isinstance(ps.tracking_margin, Margins)
                assert ps.tracking_margin == Margins(width=0.2, height=0.2)

            def test_int(self):
                ps = PartSettings(tracking_margin=1)
                assert isinstance(ps.tracking_margin, Margins)
                assert ps.tracking_margin == Margins(width=1.0, height=1.0)

            def test_dict(self):
                ps = PartSettings(
                    tracking_margin={"width": 0.2, "height": 0.2}
                )
                assert isinstance(ps.tracking_margin, Margins)
                assert ps.tracking_margin == Margins(width=0.2, height=0.2)


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
                enabled_parts=["Thing"],
                parts={
                    "Thing": {
                        "mask": "Circle",
                        "state": "protected",
                        "fade": 0.5,
                        "protection_mask": "placeholder",
                    }
                },  # type: ignore
            )
            assert len(ds.parts) == 1
            assert ds.parts["Thing"].mask == "Circle"
            assert ds.parts["Thing"].state == PartState.PROTECTED
            assert ds.parts["Thing"].fade == 0.5
            assert ds.parts["Thing"].protection_mask == "placeholder"

        def test_missing_enabled_parts(self):
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
            assert len(ds.parts) == 0

        def test_defaults(self):
            ds = DetectionSettings(
                enabled_parts=["Thing"],
                parts={"Thing": {}},  # type: ignore
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
                enabled_parts=["Thing", "Thing2"],
                parts={
                    "Thing": {
                        "mask": "Circle",
                        "state": "protected",
                        "fade": 0.5,
                        "protection_mask": "placeholder",
                    },
                    "Thing2": {},
                },  # type: ignore
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
