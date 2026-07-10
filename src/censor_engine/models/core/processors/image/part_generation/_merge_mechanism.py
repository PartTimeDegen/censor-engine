from collections import defaultdict

from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.enums import MergeMethod
from censor_engine.models.libraries.configs._helper_types import Groups

PartGroup = list[Part]
PartGroups = list[PartGroup]


class GroupingStrategy:
    """
    This class is used to hold the grouping strategies depending on the merge
    methods.

    Returns:
        PartGroups, e.g., [[Part, Part], [Part], [Part, Part, Part]]

    """

    @staticmethod
    def group_by_everything(parts: list[Part]) -> PartGroups:
        """e.g., [[Part, Part, Part, Part, Part, Part]]."""
        return [parts]

    @staticmethod
    def group_by_nothing(parts: list[Part]) -> PartGroups:
        """e.g., [[Part], [Part], [Part], [Part], [Part], [Part]]."""
        return [[part] for part in parts]

    @staticmethod
    def group_by_part_names(part_lookup: dict[str, list[Part]]) -> PartGroups:
        """
        e.g.,
        [
            [Part("a"), Part("a")],
            [Part("b")],
            [Part("c"), Part("c"), Part("c")]
        ].

        """
        return list(part_lookup.values())

    @staticmethod
    def group_by_part_groups(
        part_lookup: dict[str, list[Part]],
        groups: Groups,
    ) -> PartGroups:
        """
        e.g.,
        [
            [Part("a"), Part("a"), Part("b"), Part("b"), Part("b")],
            [Part("c")],
        ].

        with groups [
            ["a", "b"]
            ["c"]
        ]
        """
        result = []

        for group in groups:
            parts = [
                part for name in group for part in part_lookup.get(name, [])
            ]

            if parts:
                result.append(parts)

        return result


class MergeMechanismManager:
    def _create_part_lookup(self, parts: list[Part]) -> dict[str, list[Part]]:
        """
        This is a helper method to create a dictionary lookup based on the part
        name.

        Args:
            parts: List of Parts

        Returns:
            Dictionary of the part names and all the parts that have that name

        """
        lookup = defaultdict(list)
        for part in parts:
            lookup[part.get_name()].append(part)
        return lookup

    def _put_parts_into_groups(
        self, parts: list[Part], merge_method: MergeMethod, groups: Groups
    ) -> PartGroups:
        """
        This is the method that decides which strategy to use for the merge
        method.

        Args:
            parts: List of Parts
            merge_method: Merging method
            groups: List of groups provided from the config

        Returns:
            The list of parts merged into PartGroups

        """
        part_lookup = self._create_part_lookup(parts)
        match merge_method:
            case MergeMethod.ALL | MergeMethod.FULL:
                return GroupingStrategy.group_by_everything(parts)

            case MergeMethod.PARTS:
                return GroupingStrategy.group_by_part_names(part_lookup)

            case MergeMethod.GROUPS:
                return GroupingStrategy.group_by_part_groups(
                    part_lookup, groups
                )

            case _:
                return GroupingStrategy.group_by_nothing(parts)

    def _merge_groups(self, part_groups: PartGroups) -> list[Part]:
        """
        This method is used to turn the PartGroups into a list of parts again
        by reducing the groups down to the first part and merging all their
        masks.

        Args:
            part_groups: Parts Groups

        Returns:
            List of parts, now with the merged masks

        """
        parts: list[Part] = []
        for part_list in part_groups:
            part_list_by_state = sorted(
                part_list,
                key=lambda part: part.properties.settings.state,
                reverse=True,
            )

            # Skip if it's Only One Part
            if len(part_list_by_state) == 1:
                parts.append(part_list_by_state[0])
                continue

            # Add Layers of Masks to First Part
            first_part = part_list_by_state[0]
            rest_of_parts = part_list_by_state[1:]
            first_part.masks.layers_of_mask += [
                part.masks.current_mask for part in rest_of_parts
            ]

            # Compile and Add Parts to List
            first_part.masks.compile_base_masks()
            parts.append(first_part)

        return parts

    def merge_parts_based_on_merge_method(
        self,
        parts: list[Part],
        merge_method: MergeMethod,
        groups: Groups,
    ):
        """
        This method is used to merge parts based on the merge methods.

        The merge method and merging mechanism is used to combine parts and
        their masks into the first part of the list in the config's groups.

        With this mechanism, the lists give priority to parts with protected
        states such that when they are applied, the masks will apply to the
        rest of the parts it absorbed.

        Args:
            parts: List of parts
            merge_method: Merge method used
            groups: Groups of parts, from the config

        Returns:
            Merge list of parts

        """
        if merge_method == MergeMethod.NONE:
            return parts

        part_groups = self._put_parts_into_groups(
            parts=parts,
            merge_method=merge_method,
            groups=groups,
        )

        return self._merge_groups(part_groups)
