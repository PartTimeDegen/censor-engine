import os
from pathlib import Path
from typing import Callable
import progressbar
from censor_engine.models.config import Config
from censor_engine.paths import PathManager
from .mixin_pipeline_image import ImageProcessor
from .video import FrameProcessor, VideoProcessor

from .tools.debugger import DebugLevels
from .tools.dev_tools import DevTools
from .tools.video_tools import VideoInfo


class MixinVideoPipeline:
    def _make_progress_bar_widgets(
        self, index_text: str, file_name: str, total_amount: int
    ) -> list:
        return [
            f"{index_text} ",
            f'Censoring "{file_name}" > ',
            progressbar.Counter(),
            "/",
            f"{total_amount} |",
            progressbar.Percentage(),
            " [",
            progressbar.Timer(),
            "] ",
            "(",
            progressbar.ETA(),
            ") ",
            progressbar.GranularBar(),
        ]

    def _video_pipeline(
        self,
        main_files_path: str,
        indexed_files: list[tuple[int, str, str]],
        config: Config,
        debug_level: DebugLevels,
        in_place_durations: list[float],
        function_get_index: Callable[[int, int], str],
        function_display_times: Callable[[], None],
        flags: dict[str, bool],
        path_manager: PathManager,
    ):
        dev_tools = None
        max_index = max(f[0] for f in indexed_files)
        for index, file_path, file_type in indexed_files:
            # Check it's an Image
            if file_type != "video":
                continue

            # Get Video Capture
            full_file_path = path_manager.get_save_file_path(file_path)
            video_processor = VideoProcessor(file_path, full_file_path, flags)

            # Build Progress Bar
            file_name = (
                full_file_path
                if path_manager.get_flag_is_using_full_path()
                else file_path.split(os.sep)[-1]
            )

            progress_bar = progressbar.progressbar(
                range(video_processor.total_frames),
                widgets=self._make_progress_bar_widgets(
                    index_text=function_get_index(index, max_index),
                    file_name=file_name,
                    total_amount=video_processor.total_frames,
                ),
            )

            # Get Frame Capture
            frame_hold = int(
                config.video_settings.part_frame_hold_seconds
                * video_processor.get_fps()
            )

            frame_processor = FrameProcessor(
                frame_difference_threshold=config.video_settings.frame_difference_threshold,
                part_frame_hold_frames=frame_hold,
            )

            # Iterate through Frames
            frame_counter = 0
            for _ in progress_bar:
                # Check Frames
                ret, frame = video_processor.video_capture.read()
                frame_counter += 1
                if not ret:
                    break

                if flags["dev_tools"]:
                    dev_tools = DevTools(
                        output_folder=Path(file_path),
                        main_files_path=Path(main_files_path),
                        using_full_output_path=path_manager.get_flag_is_using_full_path(),
                    )

                # Run Censor Manager
                censor_manager = ImageProcessor(
                    file_image=frame,
                    config=config,
                    debug_level=debug_level,
                    dev_tools=dev_tools,
                )
                censor_manager.generate_parts_and_shapes()

                # # Apply Stability Stuff
                """
                NOTE:   This section is used to make videos more stable,
                        currently the processing effects performed are:

                            -   Holding frames for a certain number of frames
                                to avoid issues where a part doesn't get
                                detected, thus causing a flickering effect.

                            -   Maintaining the last frame instead of the 
                                current if the difference is negligible, this
                                avoids issues where the the detected areas are
                                slightly different thus causes the censors to
                                "spasm".

                """
                frame_processor.load_parts(censor_manager._image_parts)
                """
                -   Keep parts (hold them, if -1, always hold)
                -   check sizes for parts, flag any bad ones
                -   replace them with the held part
                -   if the held part is bad, update it to a better one (biggest?)
                -   
                """

                # Apply Quality Filters
                frame_processor.run()

                # Update the Parts
                # frame_processor.save_frame()
                censor_manager._image_parts = frame_processor.retrieve_parts()

                # Apply Censors
                censor_manager.compile_masks()
                censor_manager.apply_censors()

                # Save Output
                file_output = censor_manager.return_output()

                # # Apply Debug Effects
                if debug_level > DebugLevels.NONE:
                    video_info = VideoInfo(
                        frame,
                        frame_counter,
                        censor_manager._image_parts,
                        video_processor,
                        frame_processor,
                        debug_level,
                    )
                    file_output = video_info.get_debug_info(
                        file_output,
                        frame_processor,
                    )

                # Write Frame
                video_processor.write_frame(file_output)

                # Debug Show Times
                in_place_durations.append(censor_manager.get_duration())
                function_display_times()

            # Spacer
            print()
