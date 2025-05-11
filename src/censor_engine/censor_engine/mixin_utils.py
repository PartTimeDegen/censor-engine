from glob import glob
import os

from censor_engine import APPROVED_FORMATS_IMAGE, APPROVED_FORMATS_VIDEO
from censor_engine.models.config import Config


class MixinUtils:
    def _get_index_text(self, index: int, max_file_index: int):
        index_percent = index / max_file_index
        leading_spaces = " " * (len(str(max_file_index)) - len(str(index)))
        leading_spaces_pc = " " * (3 - len(str(int(100 * index_percent))))

        return f"{leading_spaces}{index}/{max_file_index} ({leading_spaces_pc}{index_percent:0.1%})"

    def _save_file(
        self,
        file_name: str,
        main_files_path: str,
        config: Config,
        force_png: bool = False,
    ) -> str:
        # Get Name
        full_path, ext = os.path.splitext(file_name)

        # Add Word Fixes
        list_path = full_path.split(os.sep)

        fixed_file_list = [
            word
            for word in [
                config.file_settings.file_prefix,
                list_path[-1],
                config.file_settings.file_suffix,
            ]
            if word != ""
        ]
        new_file_name = "_".join(fixed_file_list)

        # Get New Location
        loc_base = str(main_files_path)
        folders_loc_base = len(loc_base.split(os.sep))

        base_folder = list_path[:folders_loc_base]
        new_folder = list_path[folders_loc_base:]

        censored_folder = config.file_settings.censored_folder.split(os.sep)
        new_folder = censored_folder + new_folder[len(censored_folder) :]

        new_folder.pop()

        new_folder = base_folder + new_folder
        new_folder = os.sep.join(new_folder)

        os.makedirs(new_folder, exist_ok=True)
        file_path_new = os.sep.join([new_folder, new_file_name])

        if force_png:
            ext = ".png"

        # File Proper
        return f"{file_path_new}{ext}"

    def _find_files(
        self,
        main_files_path: str,
        config: Config,
    ):
        # Get Full Path
        full_files_path = os.path.join(
            main_files_path,
            config.file_settings.uncensored_folder,
        )
        approved_formats = APPROVED_FORMATS_IMAGE + APPROVED_FORMATS_VIDEO

        # Scan Folders
        files = [
            file_name
            for file_name in glob(
                os.path.join(full_files_path, "**", "*"), recursive=True
            )
            if os.path.isfile(file_name)
            and file_name[file_name.rfind(".") :] in approved_formats
        ]

        # Check and Filter for Approved File Formats Only
        if not files:
            raise FileNotFoundError(f"Empty folder: {full_files_path}")
        if not os.path.exists(full_files_path):
            raise TypeError(f"Path Does Not Exist: {full_files_path}")

        indexed_files = [
            (
                index,
                file_name,
                "video"
                if file_name.endswith(tuple(APPROVED_FORMATS_VIDEO))
                else "image",
            )
            for index, file_name in enumerate(files, start=1)
        ]

        if len(indexed_files) != 0:
            return indexed_files
        return [
            (
                1,
                full_files_path,
                "video"
                if full_files_path.endswith(tuple(APPROVED_FORMATS_VIDEO))
                else "image",
            )
        ]
