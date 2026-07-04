from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.enums import MergeMethod


class MergeMechanismManager:
    def _should_merge_parts(
        self,
        target: Part,
        other: Part,
        merge_method: MergeMethod,
    ) -> bool:
        if merge_method == MergeMethod.ALL:
            return True

        if merge_method == MergeMethod.FULL:
            return True

        if merge_method == MergeMethod.PARTS:
            return target.get_name() == other.get_name()

        if merge_method == MergeMethod.GROUPS:
            return other.get_name() in target.properties.group_merge

        return False

    def _put_parts_into_groups(self, parts: list[Part]): ...
    def _merge_groups(self, parts: list[Part]): ...

    def merge_parts_based_on_merge_method(
        self,
        parts: list[Part],
        merge_method: MergeMethod,
    ):
        if merge_method == MergeMethod.NONE:
            return parts

        return parts
