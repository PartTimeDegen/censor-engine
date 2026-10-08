import numpy as np
import pytest

from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.core.processors.image.part_generation._state_mechanism import (
    PartPair,
    PartStateMechanism,
    StateHandlingStrategy,
    _combine_parts,
    _subtract_masks,
)
from censor_engine.models.enums import MergeMethod, PartState

# new_part.config.image.merging.method


@pytest.fixture
def part_protected(part_base: Part) -> Part:
    new_part = part_base
    new_part.properties.settings.state = PartState.PROTECTED
    return new_part


@pytest.fixture
def part_revealed(part_base: Part) -> Part:
    new_part = part_base
    new_part.properties.settings.state = PartState.REVEALED
    return new_part


@pytest.fixture
def part_unprotected(part_base: Part) -> Part:
    new_part = part_base
    new_part.properties.settings.state = PartState.UNPROTECTED
    return new_part


@pytest.fixture
def part_list(
    part_protected: Part,
    part_revealed: Part,
    part_unprotected: Part,
) -> list[Part]:

    return [
        part_protected,
        part_protected,
        part_revealed,
        part_revealed,
        part_unprotected,
        part_unprotected,
    ]


@pytest.fixture
def part_pair(part_list: list[Part]) -> PartPair:
    return PartPair(part_list[0], part_list[4])


def test_subtract_masks(part_base: Part):
    _subtract_masks(part_base, part_base)
    assert np.all(part_base.mask_manager.current_mask == 0)


def test_combine_parts(part_list: list[Part], part_pair: PartPair):
    before_count = len(part_list)
    removed_list = []
    _combine_parts(part_pair.part_a, part_pair.part_b, part_list, removed_list)

    assert len(removed_list) == 1
    assert len(part_list) == before_count - 1
    assert np.any(part_list[0].mask_manager.current_mask != 0)


# TODO: I can't be asked to properly test this, one day I might
class TestStateHandlingStrategy:
    def test_matching_state(self, part_list: list[Part], part_pair: PartPair):
        before_count = len(part_list)
        removed_list = []
        StateHandlingStrategy.matching_state(
            part_pair, part_list, removed_list
        )

        assert len(removed_list) == 1
        assert len(part_list) == before_count - 1
        assert np.any(part_list[0].mask_manager.current_mask != 0)

    class TestProtectedState:
        def test_matching_censors(self): ...
        def test_not_matching_censors(self): ...

    class TestRevealedState:
        def test_a_is_higher(self): ...
        def test_a_is_lower(self): ...

    class TestUnprotectedState:
        def test_combine_masks(self): ...
        def test_a_has_higher_state(self): ...
        def test_neither(self): ...


class TestPartStateMechanism:
    def test_working(self, part_list: list[Part]):
        output = PartStateMechanism().handle_mask_overlaps_based_on_part_state(
            part_list
        )
        assert len(output) == 6

    def test_no_parts(self):
        output = PartStateMechanism().handle_mask_overlaps_based_on_part_state(
            []
        )

        assert output == []

    def test_merge_method_set_to_none(self, part_list: list[Part]):
        new_list = part_list
        for part in new_list:
            part.properties.config.image.merging.method == MergeMethod.NONE
        output = PartStateMechanism().handle_mask_overlaps_based_on_part_state(
            new_list
        )
        assert len(output) == 6
