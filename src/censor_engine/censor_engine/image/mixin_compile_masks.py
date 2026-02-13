from censor_engine.detected_part import Part
from censor_engine.models.enums import MergeMethod, PartState
from censor_engine.models.structs import Mixin


class MixinComponentCompile(Mixin):
    """
    This Mixin is used to handle the methods used to combine and compile masks
    based on settings provided.

    These currently are:
        -   Merge groups
        -   States

    """

    def _merge_parts(self, parts: list[Part]) -> list[Part]:
        """
        This method is used to merge parts based on their merge method and how
        the parts qualify.

        This method depends on the merge method used (see Config) and combines
        them as such. This reduces the work needed to censor an image, and
        avoids censor overlapping in some methods.

        :param list[Part] parts: The list of parts.
        :return list[Part]: The merged list of parts.
        """
        if not parts:
            return parts

        def merge_fellow_parts(
            target_part: Part,
            start_index: int,
            parts: list[Part],
            merged_indices: set[int],
            merge_method: MergeMethod,
        ) -> Part:
            """
            This function will check each part against the rest of the list as
            a closing window, then it will merge the masks if it should be
            merged.

            This function uses the list of parts then keeps tracked of merged
            parts (i.e., no longer existing) via a set `merged_indices`. The
            starting index is used to make the window smaller to avoid
            repeating parts.

            :param Part target_part: The target part
            :param int start_index: Starting Index
            :param list[Part] parts: List of parts
            :param set[int] merged_indices: Used to keep track of what parts
                have been merged.
            :param MergeMethod merge_method: The merge method used.
            :return Part: The part but with all applied merged parts, merged.
            """
            for index in range(start_index + 1, len(parts)):
                other_part = parts[index]

                # Valid Part Checks
                # NOTE: I'm aware this looks weird, technically True=1
                #       and False=0 so you can do multiplication logic to lock
                #       the value False if it gets triggered.
                is_valid_part = True

                if index in merged_indices:
                    is_valid_part = False

                match merge_method:
                    case MergeMethod.GROUPS:
                        if (
                            other_part.get_name()
                            not in target_part.merge_group
                        ):
                            is_valid_part = False

                    case MergeMethod.PARTS:
                        if other_part.get_name() != target_part.get_name():
                            is_valid_part = False

                    case MergeMethod.FULL:
                        is_valid_part = True

                if is_valid_part:
                    target_part.base_masks.extend(other_part.base_masks)
                    merged_indices.add(index)

            return target_part

        # Prep Stuff
        new_parts: list[Part] = []
        merged_indices: set[int] = set()
        merge_method = (
            parts[0].config.rendering_settings.merge_method
        )  # Assume parts all have same config

        # Method Cases
        if merge_method == MergeMethod.NONE:
            return parts

        # Iterate and Merge Parts
        for index, part in enumerate(parts):
            if index in merged_indices:
                continue

            if part.merge_group or merge_method == MergeMethod.ALL:
                part = merge_fellow_parts(  # noqa: PLW2901
                    part,
                    index,
                    parts,
                    merged_indices,
                    merge_method,
                )

                part.compile_base_masks()

            new_parts.append(part)
        return new_parts

    def _process_state_logic_for_masks(self, parts: list[Part]) -> list[Part]:  # noqa: PLR0912
        """
        This method is used to handle the part states, i.e., if the part is
        protected, unprotected, or forced to reveal the part.

        :param list[Part] parts: List of the parts
        :return list[Part]: Merged list of the parts
        """
        sorted_parts = sorted(
            parts,
            key=lambda x: (-x.part_settings.state.value, x.part_name),
            reverse=True,
        )
        removed_parts = []  # Track removed parts

        if not sorted_parts:
            return sorted_parts

        # HACK: This is is a patchwork fix, needs to properly be done
        if parts[0].config.rendering_settings.merge_method == MergeMethod.NONE:
            return sorted_parts

        for index, primary_part in enumerate(sorted_parts):
            if primary_part in removed_parts:
                continue

            primary_state = primary_part.part_settings.state

            for secondary_part in sorted_parts[index + 1 :]:
                if secondary_part in removed_parts:
                    continue

                secondary_state = secondary_part.part_settings.state

                # Quality of Life Booleans
                same_censors = (
                    primary_part.part_settings.censors
                    == secondary_part.part_settings.censors
                )
                same_state = primary_state == secondary_state
                primary_has_higher_rank = primary_state > secondary_state

                def subtract_masks(target: Part, source: Part) -> None:
                    """
                    Subtract one part's mask from another.

                    """
                    target.subtract(source.mask)

                def combine_parts() -> None:
                    """
                    Combine secondary part into primary and remove secondary.

                    """
                    primary_part.add(secondary_part.mask)  # noqa: B023
                    removed_parts.append(secondary_part)  # noqa: B023
                    parts.remove(secondary_part)  # noqa: B023

                # MATCHING: Merge if both parts have the same settings
                if same_censors and same_state:
                    combine_parts()

                # PROTECTED: If `censors` match, combine instead of subtracting
                elif PartState.PROTECTED in (primary_state, secondary_state):
                    if same_censors:
                        combine_parts()
                    elif same_state or primary_state == PartState.PROTECTED:
                        subtract_masks(secondary_part, primary_part)
                    else:
                        subtract_masks(primary_part, secondary_part)

                # REVEALED: Higher-ranked part subtracts from lower-ranked part
                elif primary_state == PartState.REVEALED:
                    subtract_masks(
                        secondary_part
                        if primary_has_higher_rank
                        else primary_part,
                        primary_part
                        if primary_has_higher_rank
                        else secondary_part,
                    )

                # UNPROTECTED: Merge if same censors, otherwise subtract
                elif primary_state == PartState.UNPROTECTED:
                    if primary_has_higher_rank and same_censors:
                        secondary_part.add(primary_part.mask)
                        removed_parts.append(primary_part)
                        parts.remove(primary_part)
                    elif primary_has_higher_rank:
                        subtract_masks(primary_part, secondary_part)

        return parts
