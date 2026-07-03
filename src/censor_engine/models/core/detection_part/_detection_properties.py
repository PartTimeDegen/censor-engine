from dataclasses import dataclass, field

from censor_engine._typing import MaskImage
from censor_engine.models.libraries.configs.settings._detection import (
    _Margins,
)
from censor_engine.models.libraries.detectors.schemas import (
    AbsoluteBBox,
    DetectorOutput,
)


@dataclass(slots=True)
class DetectionProperties:
    """
    Stores detection metadata and provides convenience accessors for
    bounding boxes, grouping information, and detector outputs.

    The original bounding box is expanded according to the configured
    margins and associated with merge and persistence groups.

    Attributes:
        detector_output (DetectorOutput): Raw detector output.

        margins (int | float | dict[str, float]): Relative bounding-box
            expansion configuration.

        groups_merge (list[list[str]]): Label groups used for merging.

        groups_persistance (list[list[str]]): Label groups used for
            persistence tracking.

    """

    detector_output: DetectorOutput

    # Config Information
    margins: _Margins
    groups_merge: list[list[str]]
    groups_persistance: list[list[str]]

    # Internal
    # # Boundary Box
    _corrected_bbox: AbsoluteBBox = field(init=False)

    # # Merge Group
    _is_merged: bool = field(init=False, default=False)
    _group_merge_id: int | None = field(init=False)
    _group_merge: list[str] = field(default_factory=list, init=False)

    # # Persistance Group
    _group_persist_id: int | None = field(init=False)
    _group_persist: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """
        Initialize derived properties and grouping metadata and computes the
        corrected bounding box, determines merge and persistence groups, and
        updates internal state flags.
        """
        # Correct Box Size to Margins
        self._corrected_bbox = self._correct_relative_box_size(
            self._original_bbox,
            self.margins,
        )

        # Get Groups
        # # Get IDs and Groups
        self._group_merge_id, self._group_merge = self._determine_group(
            name=self.label, groups=self.groups_merge
        )
        self._group_persist_id, self._group_persist = self._determine_group(
            name=self.label, groups=self.groups_persistance
        )

        # Get QoL States
        if self._group_merge_id is not None:
            self._is_merged = True

    @property
    def _original_bbox(self) -> AbsoluteBBox:
        """
        Return the original bounding box from the detector output.

        Returns:
            AbsoluteBBox: The detector bounding box.

        Raises:
            TypeError: If `detector_output.bbox` is `None`.

        """
        if self.detector_output.bbox is None:
            msg = "BBox should not be None!"
            raise TypeError(msg)
        return self.detector_output.bbox

    @property
    def _masks(self) -> list[MaskImage]:
        """
        Return the detector masks.

        Returns:
            list[MaskImage]: The detector masks.

        Raises:
            TypeError: If `detector_output.masks` is `None`.

        """
        if self.detector_output.masks is None:
            msg = "Masks should not be None!"
            raise TypeError(msg)
        return self.detector_output.masks

    @property
    def origin(self) -> str:
        """
        Return the detector origin.

        Returns:
            str: The source detector name.

        """
        return self.detector_output.origin

    @property
    def part_id(self) -> int:
        """
        Return the detector part identifier.

        Returns:
            int: The detector part identifier.

        """
        return self.detector_output.part_id

    @property
    def label(self) -> str:
        """
        Return the detection label.

        Returns:
            str: The detection label.

        Raises:
            TypeError: If `detector_output.label` is `None`.

        """
        if self.detector_output.label is None:
            msg = "Label should not be None!"
            raise TypeError(msg)
        return self.detector_output.label

    @property
    def score(self) -> float:
        """
        Return the detection confidence score.

        Returns:
            float: The detection confidence score.

        Raises:
            TypeError: If `detector_output.score` is `None`.

        """
        if self.detector_output.score is None:
            msg = "Score should not be None!"
            raise TypeError(msg)
        return self.detector_output.score

    @property
    def is_merged(self) -> bool:
        """
        Return whether the detection belongs to a merge group.

        Returns:
            bool: `True` if the detection is part of a merge group,
                otherwise `False`.

        """
        return self._is_merged

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """
        Return the corrected bounding box in XYXY format.

        Returns:
            tuple[int, int, int, int]: Bounding box as
                `(x1, y1, x2, y2)`.

        """
        return self._corrected_bbox.xyxy

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        """
        Return the corrected bounding box in XYWH format.

        Returns:
            tuple[int, int, int, int]: Bounding box as
                `(x, y, width, height)`.

        """
        return self._corrected_bbox.xywh

    def _correct_relative_box_size(
        self,
        input_bbox: AbsoluteBBox,
        margins: _Margins,
    ) -> AbsoluteBBox:
        """
        Expand a bounding box according to relative margin values.

        Margins may be provided as a single scalar applied to both
        dimensions or as a dictionary containing separate `width`
        and `height` factors.

        Args:
            input_bbox (AbsoluteBBox): Bounding box to expand.

            margins (float | dict[str, float]): Relative expansion
                factors.

        Returns:
            AbsoluteBBox: Expanded bounding box.

        """
        # Get the Margin Data Depending on Type
        w_margin = margins.width
        h_margin = margins.height

        # Get The Differences in Width and Height
        x, y, width, height = input_bbox.xywh
        dw = int(width * w_margin)
        dh = int(height * h_margin)

        return AbsoluteBBox.from_xywh(
            (x - dw // 2, y - dh // 2, width + dw, height + dh)
        )

    def _determine_group(
        self,
        name: str,
        groups: list[list[str]],
    ) -> tuple[int | None, list[str]]:
        """
        Find the group containing a given label.

        Args:
            name (str): Label to search for.

            groups (list[list[str]]): Available groups.

        Returns:
            tuple[int | None, list[str]]: The group identifier and
                matching group. Returns `(None, [])` if no match is
                found.

        """
        for index, group in enumerate(groups, start=1):
            if name in group:
                return index, group

        return None, []
