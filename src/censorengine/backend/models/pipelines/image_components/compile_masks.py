from censorengine.backend.models.structures.detected_part import Part
from censorengine.backend.models.structures.enums import PartState


class ImageComponentCompileMasks:
    def _process_state_logic_for_masks(self, parts: list[Part]):
        sorted_parts = sorted(
            parts, key=lambda x: (-x.part_settings.state, x.part_name)[::-1]
        )
        removed_parts = []  # Track removed parts

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

                def subtract_masks(target: Part, source: Part):
                    """Subtract one part’s mask from another."""
                    target.subtract(source.mask)

                def combine_parts():
                    """Combine secondary part into primary and remove secondary."""
                    primary_part.add(secondary_part.mask)
                    removed_parts.append(secondary_part)
                    parts.remove(secondary_part)

                # MATCHING: Merge if both parts have the same settings
                if same_censors and same_state:
                    combine_parts()

                # PROTECTED: If `censors` match, combine instead of subtracting
                elif (
                    primary_state == PartState.PROTECTED
                    or secondary_state == PartState.PROTECTED
                ):
                    if same_censors:
                        combine_parts()
                    elif same_state or primary_state == PartState.PROTECTED:
                        subtract_masks(secondary_part, primary_part)
                    else:
                        subtract_masks(primary_part, secondary_part)

                # REVEALED: Higher-ranked part subtracts from lower-ranked part
                elif primary_state == PartState.REVEALED:
                    subtract_masks(
                        secondary_part if primary_has_higher_rank else primary_part,
                        primary_part if primary_has_higher_rank else secondary_part,
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
