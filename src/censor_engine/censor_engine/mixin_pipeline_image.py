import cv2

from censorengine.backend.models.config import Config
from censorengine.backend.models.pipelines.base import ImageProcessor
from censorengine.backend.models.tools.debugger import DebugLevels
from censorengine.backend.models.tools.dev_tools import DevTools
from typing import Callable


class ComponentImagePipeline:
    def _image_pipeline(
        self,
        main_files_path: str,
        indexed_files: list[tuple[int, str, str]],
        config: Config,
        debug_level: DebugLevels,
        in_place_durations: list[float],
        function_get_index: Callable[[int, int], str],
        function_save_file: Callable[[str, str, Config, bool], str],
        flags: dict[str, bool],
    ):
        if not [f for f in indexed_files if f[-1] == "image"]:
            return
        for index, file_path, file_type in indexed_files:
            # Check it's an Image
            if file_type != "image":
                continue

            # Print that it's Censoring
            index_text = function_get_index(index, max([f[0] for f in indexed_files]))

            # Dev Tools
            if flags["dev_tools"]:
                dev_tools = DevTools(
                    output_folder=file_path,
                    main_files_path=main_files_path,
                    using_full_output_path=flags["show_full_output_path"],
                )
            else:
                dev_tools = None

            # Read the File
            file_image = cv2.imread(file_path)

            # Run the Censor Manager
            censor_manager = ImageProcessor(
                file_image=file_image,  # type: ignore
                config=config,
                debug_level=debug_level,
                dev_tools=dev_tools,
            )
            censor_manager.start()

            # Dev Tools
            if dev_tools:
                dev_tools.dev_decompile_masks(
                    censor_manager.image_parts,
                    subfolder="zz_complete",
                )

            # Output File Handling
            file_output = censor_manager.return_output()
            new_file_name = function_save_file(
                file_path,
                main_files_path,
                config,
                censor_manager.force_png,
            )

            # File Save
            cv2.imwrite(new_file_name, file_output)

            # Print Out
            prefix = ""
            if not flags["show_full_output_path"]:
                prefix = "./"
                new_file_name = new_file_name.replace(main_files_path, "", 1)[1:]
            print(f'{index_text} Censored: "{prefix}{new_file_name}"')

            # Save Duration
            in_place_durations.append(censor_manager.get_duration())
            if flags["pad_individual_items"]:
                print()

        print("Finished Censoring Images!")
