import uuid
from collections import defaultdict

import pytest

# Replace these imports with your actual test factories/classes
from censor_engine.models.core.detection_part.detection_part import Part
from censor_engine.models.core.processors.image.part_generation._merge_mechanism import (
    GroupingStrategy,
    MergeMechanismManager,
)
from censor_engine.models.enums import MergeMethod
from censor_engine.models.libraries.configs._helper_types import Groups
from censor_engine.models.libraries.configs.config import Config
from censor_engine.models.libraries.detectors.schemas import DetectorOutput


@pytest.fixture
def parts_list():
    names = [
        "boobs",
        "boobs",
        "boobs",
        "tits",
        "tits",
        "tits",
        "breasts",
        "breasts",
        "bum",
        "bum",
        "bum",
        "bum",
        "bum",
        "toes",
        "hands",
    ]
    config = Config.from_dict(
        {"detection": {"enabled_parts": list(set(names))}}
    )
    print(config.detection.parts)

    return [
        Part(
            detector_output=DetectorOutput(
                origin="placeholder", part_id=index, score=0.2, label=name
            ),
            config=config,
            file_uuid=uuid.uuid4(),
            image_shape=(500, 500),
        )
        for index, name in enumerate(names)
    ]


@pytest.fixture
def groups():
    return [
        ["boobs", "tits"],
        ["breasts", "toes"],
    ]


# class TestGroupingStrategy:
#     def test_group_by_everything(self):
#         parts = [Part("a"), Part("b")]

#         result = GroupingStrategy.group_by_everything(parts)

#         assert result == [parts]

#     def test_group_by_everything_empty(self):
#         result = GroupingStrategy.group_by_everything([])

#         assert result == [[]]

#     def test_group_by_nothing(self):
#         parts = [
#             Part("a"),
#             Part("b"),
#         ]

#         result = GroupingStrategy.group_by_nothing(parts)

#         assert result == [
#             [parts[0]],
#             [parts[1]],
#         ]

#     def test_group_by_nothing_empty(self):
#         assert GroupingStrategy.group_by_nothing([]) == []

#     def test_group_by_part_names(self):
#         lookup = {
#             "a": [
#                 Part("a"),
#                 Part("a"),
#             ],
#             "b": [
#                 Part("b"),
#             ],
#         }

#         result = GroupingStrategy.group_by_part_names(lookup)

#         assert len(result) == 2
#         assert len(result[0]) == 2
#         assert len(result[1]) == 1

#     def test_group_by_part_groups(self):
#         lookup = {
#             "a": [
#                 Part("a"),
#             ],
#             "b": [
#                 Part("b"),
#                 Part("b"),
#             ],
#             "c": [
#                 Part("c"),
#             ],
#         }

#         groups = [
#             ["a", "b"],
#             ["c"],
#         ]

#         result = GroupingStrategy.group_by_part_groups(
#             lookup,
#             groups,
#         )

#         assert len(result) == 2
#         assert len(result[0]) == 3
#         assert len(result[1]) == 1

#     def test_group_by_part_groups_ignores_missing_parts(self):
#         lookup = {
#             "a": [
#                 Part("a"),
#             ],
#         }

#         groups = [
#             ["missing"],
#             ["a"],
#         ]

#         result = GroupingStrategy.group_by_part_groups(
#             lookup,
#             groups,
#         )

#         assert len(result) == 1
#         assert result[0][0].get_name() == "a"


class TestMergeMechanismManager:
    def test_initiate(self):
        mm = MergeMechanismManager()

    class TestMergePartsBasedOnMergeMethod:
        def test_baseline(
            self,
            parts_list: list[Part],
            groups: Groups,
        ):
            mm = MergeMechanismManager()
            output = mm.merge_parts_based_on_merge_method(
                parts_list,
                MergeMethod.PARTS,
                groups,
            )


#     def test_create_part_lookup(
#         self,
#         merge_manager,
#         part_list,
#     ):
#         result = merge_manager._create_part_lookup(part_list)

#         assert isinstance(result, defaultdict)
#         assert len(result["a"]) == 2
#         assert len(result["b"]) == 1
#         assert len(result["c"]) == 1

#     def test_create_part_lookup_empty(self, merge_manager):
#         result = merge_manager._create_part_lookup([])

#         assert result == {}

#     @pytest.mark.parametrize(
#         "merge_method, expected_groups",
#         [
#             (MergeMethod.ALL, 1),
#             (MergeMethod.FULL, 1),
#             (MergeMethod.PARTS, 3),
#             (MergeMethod.GROUPS, 2),
#             (MergeMethod.NONE, 4),
#         ],
#     )
#     def test_put_parts_into_groups(
#         self,
#         merge_manager,
#         part_list,
#         groups,
#         merge_method,
#         expected_groups,
#     ):
#         result = merge_manager._put_parts_into_groups(
#             part_list,
#             merge_method,
#             groups,
#         )

#         assert len(result) == expected_groups

#     def test_put_parts_into_groups_unknown_defaults_to_nothing(
#         self,
#         merge_manager,
#         part_list,
#         groups,
#     ):
#         result = merge_manager._put_parts_into_groups(
#             part_list,
#             "unknown",
#             groups,
#         )

#         assert len(result) == len(part_list)

#     def test_merge_groups_single_parts(
#         self,
#         merge_manager,
#     ):
#         parts = [
#             [Part("a")],
#             [Part("b")],
#         ]

#         result = merge_manager._merge_groups(parts)

#         assert len(result) == 2
#         assert result[0].get_name() == "a"

#     def test_merge_groups_combines_parts(
#         self,
#         merge_manager,
#     ):
#         first = Part("a")
#         second = Part("a")

#         result = merge_manager._merge_groups(
#             [
#                 [
#                     first,
#                     second,
#                 ]
#             ]
#         )

#         assert len(result) == 1
#         assert result[0] == first

#     def test_merge_groups_empty(self, merge_manager):
#         result = merge_manager._merge_groups([])

#         assert result == []


# class TestMergeWorkflow:
#     def test_none_returns_original_parts(
#         self,
#         merge_manager,
#         part_list,
#         groups,
#     ):
#         result = merge_manager.merge_parts_based_on_merge_method(
#             part_list,
#             MergeMethod.NONE,
#             groups,
#         )

#         assert result is part_list

#     def test_all_merges_everything(
#         self,
#         merge_manager,
#         part_list,
#         groups,
#     ):
#         result = merge_manager.merge_parts_based_on_merge_method(
#             part_list,
#             MergeMethod.ALL,
#             groups,
#         )

#         assert len(result) == 1

#     def test_full_merges_everything(
#         self,
#         merge_manager,
#         part_list,
#         groups,
#     ):
#         result = merge_manager.merge_parts_based_on_merge_method(
#             part_list,
#             MergeMethod.FULL,
#             groups,
#         )

#         assert len(result) == 1

#     def test_parts_merges_same_named_parts(
#         self,
#         merge_manager,
#         part_list,
#         groups,
#     ):
#         result = merge_manager.merge_parts_based_on_merge_method(
#             part_list,
#             MergeMethod.PARTS,
#             groups,
#         )

#         names = [part.get_name() for part in result]

#         assert names == [
#             "a",
#             "b",
#             "c",
#         ]

#     def test_groups_merges_configured_groups(
#         self,
#         merge_manager,
#         part_list,
#         groups,
#     ):
#         result = merge_manager.merge_parts_based_on_merge_method(
#             part_list,
#             MergeMethod.GROUPS,
#             groups,
#         )

#         assert len(result) == 2


# class TestMergeEdgeCases:
#     def test_empty_parts(
#         self,
#         merge_manager,
#         groups,
#     ):
#         result = merge_manager.merge_parts_based_on_merge_method(
#             [],
#             MergeMethod.ALL,
#             groups,
#         )

#         assert result == [[]]

#     def test_groups_with_no_matching_parts(
#         self,
#         merge_manager,
#     ):
#         result = merge_manager.merge_parts_based_on_merge_method(
#             [
#                 Part("a"),
#             ],
#             MergeMethod.GROUPS,
#             [
#                 ["missing"],
#             ],
#         )

#         assert result == []

#     def test_duplicate_part_names(
#         self,
#         merge_manager,
#     ):
#         parts = [
#             Part("same"),
#             Part("same"),
#             Part("same"),
#         ]

#         result = merge_manager.merge_parts_based_on_merge_method(
#             parts,
#             MergeMethod.PARTS,
#             [],
#         )

#         assert len(result) == 1

#     def test_groups_order_is_preserved(
#         self,
#         merge_manager,
#     ):
#         parts = [
#             Part("b"),
#             Part("a"),
#         ]

#         result = merge_manager.merge_parts_based_on_merge_method(
#             parts,
#             MergeMethod.GROUPS,
#             [
#                 ["a"],
#                 ["b"],
#             ],
#         )

#         assert result[0].get_name() == "a"
