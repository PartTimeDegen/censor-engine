from collections.abc import Callable
from pathlib import Path

import cv2

from censor_engine.censor_engine.tools.config_previewer.base import (
    get_config_preview,
)
from censor_engine.models.caching.base import Cache
from censor_engine.models.config import Config
from censor_engine.models.lib_models.detectors import DetectedPartSchema
from censor_engine.models.structs import Mixin
from censor_engine.paths import PathManager
from censor_engine.typing import Image

from .image import ImageProcessor
from .tools.debugger import DebugLevels
from .tools.dev_tools import DevTools


class MixinImagePipeline(Mixin):
    def _image_pipeline(
        self,
        main_files_path: Path,
        indexed_files: list[tuple[int, str, str]],
        config: Config,
        debug_level: DebugLevels,
        in_place_durations: list[float],
        function_get_index: Callable[[int, int], str],
        flags: dict[str, bool],
        path_manager: PathManager,
        *,
        frame: int = 0,
        inline_mode: bool = False,
        _test_detection_output: list[DetectedPartSchema] | None = None,
    ) -> list[Image]:
        filter_for_imaged = [f for f in indexed_files if f[-1] == "image"]
        if not path_manager.test_mode and not filter_for_imaged:
            return []

        in_memory_files: list[Image] = []  # Currently Only Test Mode
        for index, file_path, file_type in indexed_files:
            # Check it's an Image
            if file_type not in {"image", "preview"}:
                continue

            # Print that it's Censoring
            function_get_index(
                index,
                max([f[0] for f in indexed_files]),
            )

            # Dev Tools
            dev_tools = None

            if file_type != "preview":
                if flags["dev_tools"]:
                    dev_tools = DevTools(
                        output_folder=Path(file_path),
                        main_files_path=Path(main_files_path),
                        using_full_output_path=flags["show_full_output_path"],
                    )

                # Read the File
                file_image: Image = cv2.imread(file_path)

                # Caching
                cache = Cache(
                    path_manager.get_cache_folder(),
                    path_manager.base_directory,
                    file_path,
                    is_video=False,
                )
            else:
                config_info = get_config_preview(
                    config.censor_settings.enabled_parts
                )

                file_image = config_info["preview"]
                _test_detection_output = config_info["detection_data"]
                cache = None

            # Run the Censor Manager
            image_processor = ImageProcessor(
                file_image=file_image,
                file_name=file_path,
                path_manager=path_manager,
                cache=cache,
                config=config,
                debug_level=debug_level,
                dev_tools=dev_tools,
                _test_detection_output=_test_detection_output,
            )
            image_processor.start()

            # Dev Tools
            if dev_tools:
                dev_tools.dev_decompile_masks(
                    image_processor.get_image_parts(),
                    subfolder="zz_complete",
                )

            # Output File Handling
            file_output = image_processor.return_output()
            new_file_name = path_manager.get_save_file_path(
                file_path,
                force_png=image_processor.force_png,
            )

            # File Save
            print(new_file_name)  # noqa: T201
            cv2.imwrite(new_file_name, file_output)
            if inline_mode:
                in_memory_files.append(file_output)

            # Print Out
            if not flags["show_full_output_path"]:
                new_file_name = new_file_name.replace(
                    str(main_files_path),
                    "",
                    1,
                )[1:]

            # Save Duration
            if flags["pad_individual_items"]:
                pass

        return in_memory_files
