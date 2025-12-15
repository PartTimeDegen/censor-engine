from dataclasses import dataclass, field

from censor_engine.detected_part import Part


@dataclass(slots=True)
class TrackedPart:
    part: Part
    track_id: int

    # Creation Debug Info
    s_area: bool = False
    s_merge: bool = False
    s_persistence: bool = False
    s_type: bool = False

    # Qualifications
    box: tuple[int, int, int, int] = field(init=False)
    merge_group_id: int | None = field(init=False)
    part_class: str = field(init=False)

    # Internal
    misses: int = 0

    def __post_init__(self):
        self.box = self.part.relative_box
        self.merge_group_id = self.part.merge_group_id
        self.part_class = self.part.get_name()


@dataclass(slots=True)
class Tracker:
    max_missed: int

    # Internal
    _next_track_id: int = 0
    _tracked_parts: list[TrackedPart] = field(default_factory=list)

    # Tested Features
    def __compare_area(
        self, tracked_part: TrackedPart, candidate_part: Part
    ) -> bool:
        return tracked_part.part.part_area.check_in_approx_region(
            candidate_part.part_area.region
        )

    def __compare_merge_group(
        self, tracked_part: TrackedPart, candidate_part: Part
    ) -> bool:
        return tracked_part.merge_group_id == candidate_part.merge_group_id

    def __compare_persistence_group(
        self, tracked_part: TrackedPart, candidate_part: Part
    ) -> bool:
        return (
            tracked_part.part.persistence_group_id
            == candidate_part.persistence_group_id
        )

    def __compare_part_class(
        self, tracked_part: TrackedPart, candidate_part: Part
    ) -> bool:
        return tracked_part.part_class == candidate_part.get_name()

    # Public Methods
    def update_tracker(self, list_of_parts: list[Part]):
        # Temp Variables
        updated_indices: list[int] = []

        # Iteration
        print(" === Loop")

        # # First Frame Handling
        if self._next_track_id == 0:
            for part in list_of_parts:
                print(f"{part}")
                self._tracked_parts.append(
                    TrackedPart(part=part, track_id=self._next_track_id)
                )
                self._next_track_id += 1
            return

        for part in list_of_parts:
            print(f"{part}")
            is_part_found = False
            addition_indices: list[int] = []
            for index, tracked_part in enumerate(self._tracked_parts):
                # Checks
                same_area = self.__compare_area(tracked_part, part)
                same_type = self.__compare_part_class(tracked_part, part)

                same_merge_group = self.__compare_merge_group(
                    tracked_part, part
                )
                same_persistence_group = self.__compare_persistence_group(
                    tracked_part, part
                )

                # Confirmation
                same_common = same_area
                same_combined_part = (
                    same_type or same_merge_group or same_persistence_group
                )
                same_part = same_common and same_combined_part

                # Found Part
                # NOTE: This should break but it needs to do cleanup for any
                #       other parts
                if same_part and is_part_found:
                    addition_indices.append(index)
                if same_part:
                    print("- Found:", tracked_part)
                    self._tracked_parts[index] = TrackedPart(
                        part=part,
                        track_id=tracked_part.track_id,
                        s_area=same_area,
                        s_merge=same_merge_group,
                        s_type=same_type,
                    )
                    updated_indices.append(index)
                    is_part_found = True

            # Check if Part Not Found
            if not is_part_found:
                self._tracked_parts.append(
                    TrackedPart(part=part, track_id=self._next_track_id)
                )
                # NOTE: Avoid Aging New Parts
                updated_indices.append(self._next_track_id)
                self._next_track_id += 1

            # Remove Duplicates
            for index in addition_indices:
                self._tracked_parts = [
                    tracked_part
                    for index, tracked_part in enumerate(self._tracked_parts)
                    if index not in addition_indices
                ]

        print("Len:", len(self._tracked_parts))

        # Update Misses
        for index, tracked_part in enumerate(self._tracked_parts):
            if index in updated_indices:
                continue

            tracked_part.misses += 1

        # Remove Expired Parts
        self._tracked_parts = [
            tracked_part
            for tracked_part in self._tracked_parts
            if tracked_part.misses <= self.max_missed
        ]
        print(*self._tracked_parts, sep="\n")
        print(" === Loop")
        print()

    def get_parts(self) -> list[Part]:
        return [tracked_part.part for tracked_part in self._tracked_parts]
