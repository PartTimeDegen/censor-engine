from dataclasses import dataclass

from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.enums import MergeMethod, PartState


@dataclass(slots=True)
class PartPair:
    part_a: Part
    part_b: Part

    @property
    def state_a(self) -> PartState:
        return self.part_a.properties.settings.state

    @property
    def state_b(self) -> PartState:
        return self.part_b.properties.settings.state

    @property
    def matching_censors(self) -> bool:
        return (
            self.part_a.properties.settings.censors
            == self.part_b.properties.settings.censors
        )

    @property
    def matching_state(self) -> bool:
        return self.state_a == self.state_b

    @property
    def a_has_higher_state(self) -> bool:
        return self.state_a > self.state_b

    @property
    def arg_order(self) -> tuple[Part, Part]:
        return (self.part_a, self.part_b)

    @property
    def inverse_arg_order(self) -> tuple[Part, Part]:
        return (self.part_b, self.part_a)


def _subtract_masks(target: Part, source: Part) -> None:
    target.mask_manager.subtract_from_current_mask(
        source.mask_manager.current_mask
    )


def _combine_parts(
    target_part: Part,
    subject_part: Part,
    parts: list[Part],
    removed_parts: list[Part],
) -> None:
    target_part.mask_manager.add_to_current_mask(
        subject_part.mask_manager.current_mask
    )
    removed_parts.append(subject_part)
    parts.remove(subject_part)


class StateHandlingStrategy:
    @staticmethod
    def matching_state(
        part_pair: PartPair,
        parts: list[Part],
        removed_parts: list[Part],
    ) -> None:
        _combine_parts(
            part_pair.part_a,
            part_pair.part_b,
            parts,
            removed_parts,
        )

    @staticmethod
    def protected_state(
        part_pair: PartPair,
        parts: list[Part],
        removed_parts: list[Part],
    ) -> None:
        if part_pair.matching_censors:
            _combine_parts(
                part_pair.part_a,
                part_pair.part_b,
                parts,
                removed_parts,
            )
        else:
            # NOTE: This is to avoid overlapping, where the protected
            #       part won't be affected thus the non-protected
            #       (or B if both are protected) submits
            #       and gets its mask subtracted.
            protected_part_order = (
                part_pair.inverse_arg_order
                if part_pair.matching_state
                or part_pair.state_a == PartState.PROTECTED
                else part_pair.arg_order
            )
            _subtract_masks(*protected_part_order)

    @staticmethod
    def revealed_state(part_pair: PartPair) -> None:
        part_order = (
            part_pair.arg_order
            if part_pair.a_has_higher_state
            else part_pair.inverse_arg_order
        )
        _subtract_masks(*part_order)

    @staticmethod
    def unprotected_state(
        part_pair: PartPair, parts: list[Part], removed_parts: list[Part]
    ) -> None:
        if part_pair.a_has_higher_state and part_pair.matching_censors:
            _combine_parts(
                part_pair.part_a,
                part_pair.part_b,
                parts,
                removed_parts,
            )

        elif part_pair.a_has_higher_state:
            _subtract_masks(*part_pair.arg_order)


class PartStateMechanism:
    def handle_mask_overlaps_based_on_part_state(
        self, parts: list[Part]
    ) -> list[Part]:
        # Sort Parts based off State then Name
        sorted_parts = sorted(
            parts,
            key=lambda x: (x.properties.settings.state.value, x.get_name()),
            reverse=True,
        )

        # NOTE: This is used to track parts that have been merged
        removed_parts: list[Part] = []

        # Handle Empty Lists
        if not sorted_parts:
            return sorted_parts

        # Prematurely Return if the Merge Method is None
        # NOTE: Merge method is global so this works, change if it gets made to
        #       be per part.
        if parts[0].config.image.merging.method == MergeMethod.NONE:
            return sorted_parts

        # Merging Algorithm
        for index, part_a in enumerate(sorted_parts):
            if part_a in removed_parts:
                continue

            for part_b in sorted_parts[index + 1 :]:
                if part_b in removed_parts:
                    continue

                part_pair = PartPair(part_a, part_b)

                # MATCHING: Merge if both parts have the same settings
                if part_pair.matching_censors and part_pair.matching_state:
                    StateHandlingStrategy.matching_state(
                        part_pair, parts, removed_parts
                    )

                # PROTECTED: If `censors` match, combine instead of subtracting
                elif PartState.PROTECTED in (
                    part_pair.state_a,
                    part_pair.state_b,
                ):
                    StateHandlingStrategy.protected_state(
                        part_pair, parts, removed_parts
                    )

                # REVEALED: Higher-ranked part subtracts from lower-ranked part
                elif part_pair.state_a == PartState.REVEALED:
                    StateHandlingStrategy.revealed_state(part_pair)

                # UNPROTECTED: Merge if same censors, otherwise subtract
                elif part_pair.state_a == PartState.UNPROTECTED:
                    StateHandlingStrategy.unprotected_state(
                        part_pair, parts, removed_parts
                    )

        return parts
