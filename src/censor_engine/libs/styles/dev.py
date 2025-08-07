# type: ignore
# import cv2
# import numpy as np

# from censor_engine.models.structs.colours import Colour, _colours
# from censor_engine.libs.detectors.multi_detectors import NudeNetDetector
# from censor_engine.libs.registries import StyleRegistry
# from censor_engine.models.lib_models.styles import DevStyle

# from censor_engine.models.structs.contours import Contour
# from censor_engine.typing import Image, Mask

# from censor_engine.detected_part import Part


# colour_dict = {
#     part_name: colour
#     for part_name, colour in zip(
#         NudeNetDetector.model_classifiers,
#         _colours.keys(),
#     )
# }


# # @StyleRegistry.register()
# # class Debug(DevStyle):  # TODO
#     def _apply_mask_as_overlay(
#         self,
#         image: Image,
#         mask: Mask,
#         colour: Colour,
#         alpha: float,
#     ) -> Image:
#         overlay = image.copy()

#         # Create a single-channel boolean mask from any RGB mask channel
#         mask_bool = mask[:, :, 0] > 0

#         if not np.any(mask_bool):
#             return overlay  # Nothing to do if mask is empty

#         # Create an array of shape (H, W, 3) with the target color
#         color_array = np.full_like(image, colour.value, dtype=image.dtype)

#         # Alpha blending only on masked region
#         if alpha < 1.0:
#             # Blend only in masked region
#             overlay[mask_bool] = (
#                 (1 - alpha) * image[mask_bool] + alpha * color_array[mask_bool]
#             ).astype(image.dtype)
#         else:
#             # Hard color replace in masked region
#             overlay[mask_bool] = color_array[mask_bool]

#         return overlay

#     def apply_style(  # TOOD: the main problem I'm having is that I can't access deleted parts
#         self,
#         image: Image,
#         mask: Mask,
#         contours: list[Contour],
#         part: Part,
#         part_list: list[Part],
#     ) -> Image:
#         for part_obj in part_list:
#             colour_obj = Colour(colour_dict[part.get_name()])
#             image = self._apply_mask_as_overlay(image, mask, colour_obj, 0.2)

#             contours_points = [contour.points for contour in contours]
#             cv2.drawContours(
#                 overlay,
#                 contours_points,
#                 -1,
#                 colour_obj.value,
#                 thickness=1,
#                 lineType=self.default_linetype,
#             )

#             # Inputs
#             text = f"{part.part_name}\nSCORE={float(part.score):0.2%}"
#             coords = (
#                 part.relative_box[0],
#                 part.relative_box[1] + part.relative_box[3],
#             )

#         # Actions

#         return overlay
