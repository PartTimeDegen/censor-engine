from dataclasses import dataclass, field

from censor_engine._typing import MaskImage
from censor_engine.models.libraries.configs._helper_types import (
    BoundPercentage,
    Groups,
)
from censor_engine.models.libraries.configs.config import Config
from censor_engine.models.libraries.configs.settings._detection import (
    PartSettings,
)
from censor_engine.models.libraries.detectors.schemas import (
    AbsoluteBBox,
    DetectorOutput,
)


@dataclass(slots=True)
class PartProperties:
    """
    TODO.

    Attributes:
        detector_output (DetectorOutput): Raw detector output.

        margins (int | float | dict[str, float]): Relative bounding-box
            expansion configuration.

        groups_merge (list[list[str]]): Label groups used for merging.

        groups_persistance (list[list[str]]): Label groups used for
            persistence tracking.

    """

    detector_output: DetectorOutput
    config: Config

    # Internal
    # # Part Config
    settings: PartSettings = field(init=False)

    # # Boundary Box
    bbox: AbsoluteBBox | None = field(init=False, default=None)

    # Public Stuff
    # # Detector Outputs
    detector_origin: str = field(init=False)
    part_id: int = field(init=False)
    label: str = field(init=False)
    score: BoundPercentage = field(init=False)
    original_bbox: AbsoluteBBox | None = field(init=False)
    masks: list[MaskImage] | None = field(init=False)

    # # Merge Group
    is_merged: bool = field(init=False, default=False)
    group_merge_id: int | None = field(init=False)
    group_merge: list[str] = field(default_factory=list, init=False)

    # # Persistance Group
    group_persist_id: int | None = field(init=False)
    group_persist: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """
        Initialize derived properties and grouping metadata and computes the
        corrected bounding box, determines merge and persistence groups, and
        updates internal state flags.
        """
        self._load_detector_data()
        self._load_config_data()

        # Correct Box Size to Margins
        if self.original_bbox is not None:
            self.bbox = self.original_bbox.rescale_bbox(self.settings.margins)

        # Get QoL States
        self.is_merged = self.group_merge_id is not None

    # Methods
    def _load_detector_data(self) -> None:
        if self.detector_output.label is None:
            msg = "Label should not be None"
            raise TypeError(msg)
        if self.detector_output.score is None:
            msg = "Score should not be None"
            raise TypeError(msg)

        self.detector_origin = self.detector_output.origin
        self.part_id = self.detector_output.part_id
        self.score = self.detector_output.score
        self.label = self.detector_output.label
        self.original_bbox = self.detector_output.bbox
        self.masks = self.detector_output.masks

    def _load_config_data(self) -> None:
        # Part Config
        self.settings = self.config.detection.parts[self.label]

        # Get Groups
        group_infos = self.config.groups
        self.group_merge_id, self.group_merge = self._determine_group(
            name=self.label, groups=group_infos.merging
        )
        self.group_persist_id, self.group_persist = self._determine_group(
            name=self.label, groups=group_infos.persistance
        )

    def _determine_group(
        self,
        name: str,
        groups: Groups,
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
