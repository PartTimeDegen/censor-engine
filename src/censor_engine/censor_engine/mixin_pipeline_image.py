from collections.abc import Callable
from pathlib import Path

import cv2

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
        inline_mode: bool,
        _test_detection_output: list[DetectedPartSchema] | None,
    ) -> list[Image]:
        if not [f for f in indexed_files if f[-1] == "image"]:
            return []

        in_memory_files: list[Image] = []  # Currently Only Test Mode
        for index, file_path, file_type in indexed_files:
            # Check it's an Image
            if file_type != "image":
                continue

            # Print that it's Censoring
            index_text = function_get_index(
                index, max([f[0] for f in indexed_files])
            )

            # Dev Tools
            dev_tools = None
            if flags["dev_tools"]:
                dev_tools = DevTools(
                    output_folder=Path(file_path),
                    main_files_path=Path(main_files_path),
                    using_full_output_path=flags["show_full_output_path"],
                )

            # Read the File
            file_image = cv2.imread(file_path)

            # Run the Censor Manager
            image_processor = ImageProcessor(
                file_image=file_image,  # type: ignore
                config=config,
                debug_level=debug_level,
                dev_tools=dev_tools,
                _test_detection_output=_test_detection_output,
            )
            image_processor.start()

            # Dev Tools
            if dev_tools:
                dev_tools.dev_decompile_masks(
                    image_processor._image_parts,
                    subfolder="zz_complete",
                )

            # Output File Handling
            file_output = image_processor.return_output()
            new_file_name = path_manager.get_save_file_path(
                file_path, image_processor._force_png
            )

            # File Save
            cv2.imwrite(new_file_name, file_output)
            if inline_mode:
                in_memory_files.append(file_output)

            # Print Out
            prefix = ""
            if not flags["show_full_output_path"]:
                prefix = "./"
                new_file_name = new_file_name.replace(
                    str(main_files_path), "", 1
                )[1:]
            print(f'{index_text} Censored: "{prefix}{new_file_name}"')

            # Save Duration
            in_place_durations.append(image_processor.get_duration())
            if flags["pad_individual_items"]:
                print()

        print("Finished Censoring Images!")
        return in_memory_files
