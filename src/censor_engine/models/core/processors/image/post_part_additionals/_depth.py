import cv2
import numpy as np

from censor_engine.models.core.detection_part._additionals_schema import (
    DepthProperties,
)
from censor_engine.models.core.detection_part.detection_part import (
    Part,
    PartList,
)
from censor_engine.models.core.tools.mask_helpers import (
    edge_smoothing,
    hole_filling,
    island_cleanup,
)
from censor_engine.models.libraries.configs.config import Config
from censor_engine.models.libraries.configs.settings._engine_settings import (
    _DepthSettings,
)
from censor_engine.typing import DepthImage, MaskImage


class DepthManager:
    def _apply_depth_values(self, part: Part, depth_map: DepthImage) -> Part:
        x, y = part.properties.bbox.centre
        depth_value = depth_map[y, x]
        part.properties.additionals.depth = DepthProperties(
            centroid_depth=depth_value,
            censor_layer=0,
        )
        return part

    def _get_band_map(
        self,
        people_mask: MaskImage,
        layers: int,
        depth_map: DepthImage,
        band_tune: float = 1.10,
    ) -> DepthImage:
        # Apply Mask to Depth
        filtered_depth = np.where(people_mask, depth_map, np.nan)
        valid_values = filtered_depth[np.isfinite(filtered_depth)]

        # Calculate Outliers to Remove Outlier Values
        q1 = np.percentile(valid_values, 25)
        q3 = np.percentile(valid_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        cleaned_depth = np.where(
            (filtered_depth >= lower_bound) & (filtered_depth <= upper_bound),
            filtered_depth,
            np.nan,
        )

        # Quantise Mask
        valid_mask = np.isfinite(cleaned_depth)
        valid_depths = cleaned_depth[valid_mask]
        min_depth = np.nanmin(valid_depths)
        max_depth = np.nanmax(valid_depths)

        # Normalise Values to found Values
        scaled = (
            (cleaned_depth[valid_mask] - min_depth)
            / (max_depth - min_depth)
            * layers
        )

        # Quantise the Values
        band_indices = np.floor(scaled).astype(np.int32)
        band_indices = band_indices * band_tune
        band_indices = np.clip(band_indices, 0, layers - 1)

        quantised = np.zeros(filtered_depth.shape, dtype=np.uint8)
        quantised[valid_mask] = band_indices.astype(np.uint8)

        return quantised.astype(np.float32)  # type: ignore

    def apply_depth_layers(
        self,
        parts_list: PartList,
        config: Config,
    ) -> PartList:
        # Depth Map
        if (
            parts_list.additionals.depth is None
            or parts_list.additionals.depth.depth_map is None
            or config.engine_settings.extra_ai_model_features.depth is None
        ):
            return parts_list

        # Shortcut Variables & Type Narrowing
        depth_settings = config.engine_settings.extra_ai_model_features.depth
        if not isinstance(depth_settings, _DepthSettings):
            msg = "Depth Settings didn't convert"
            raise TypeError(msg)

        list_of_parts = parts_list.parts

        depth_map = parts_list.additionals.depth.depth_map
        people_mask = parts_list.additionals.people_mask
        if people_mask is None:
            msg_0 = "People Mask is Missing"
            raise TypeError(msg_0)

        # Handle Dynamic Part Depth
        depth_settings.banding = True
        layers = depth_settings.layers
        if layers == 0:
            layers = len(list_of_parts)
        if depth_settings.banding:
            layers *= 2

        # Create DepthPositionHelper for Parts
        kern = 7
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern, kern))
        shrunk_people_mask = cv2.erode(people_mask, kernel, iterations=1)
        banding_map = self._get_band_map(
            layers=layers,
            depth_map=depth_map,
            people_mask=shrunk_people_mask,
        )

        map_used = banding_map if depth_settings.banding else depth_map

        # Apply Depth and Calculate Censor Layers
        parts_list.parts = sorted(
            [
                self._apply_depth_values(part, map_used)
                for part in list_of_parts
            ],
            key=lambda part: part.properties.additionals.depth.centroid_depth,  # type: ignore
            reverse=not depth_settings.banding,
        )
        number_parts = len(parts_list.parts) - 1
        for index, part in enumerate(parts_list.parts):
            if part.properties.additionals.depth is None:
                msg_1 = "Missing depth part additional"
                raise TypeError(msg_1)

            part.properties.additionals.depth.censor_layer = (
                int(part.properties.additionals.depth.centroid_depth) - 1
                if depth_settings.banding
                else number_parts - index
            )

        # Sort List to Back to Front and Change Depth Map
        parts_list.additionals.depth.depth_map = map_used
        parts_list.parts = sorted(
            parts_list.parts,
            key=lambda part: part.properties.additionals.depth.censor_layer,  # type: ignore
            reverse=True,
        )

        for part in parts_list.parts:
            if part.properties.additionals.depth is None:
                msg_2 = "Missing Depth"
                raise TypeError(msg_2)
            target_depth = part.properties.additionals.depth.centroid_depth

            filtered_mask = (
                np.where((map_used >= target_depth), map_used, 0) * 255
            ).astype(np.uint8)

            cleaned_pre_mask = cv2.dilate(
                hole_filling.fill_holes_within_mask(
                    edge_smoothing.remove_stringy_parts_of_mask(
                        filtered_mask, kernal_size=15
                    )
                ),
                kernel,
                iterations=1,
            )
            mask = cv2.subtract(
                shrunk_people_mask.astype(np.uint8), cleaned_pre_mask
            )
            cleaned_mask = island_cleanup.filter_to_largest_connected_islands(
                edge_smoothing.remove_stringy_parts_of_mask(mask)  # type: ignore
            )

            part.mask_manager.current_mask = cv2.add(
                cv2.subtract(part.mask_manager.current_mask, cleaned_mask),
                part.mask_manager.mask_obj.generate_base_mask(
                    part.mask_manager.mask_context
                ),
            )  # type: ignore

        return parts_list
