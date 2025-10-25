from censor_engine.models.structs import Mixin
from censor_engine.paths import PathManager

APPROVED_FORMATS_IMAGE = [".jpg", ".jpeg", ".png", ".webp"]
APPROVED_FORMATS_VIDEO = [".mp4", ".webm"]


class MixinUtils(Mixin):
    """
    This Mixin is used to hold the misc/utils functions.

    """

    def _get_index_text(self, index: int, max_file_index: int) -> str:
        index_percent = index / max_file_index
        leading_spaces = " " * (len(str(max_file_index)) - len(str(index)))
        leading_spaces_pc = " " * (3 - len(str(int(100 * index_percent))))

        return (
            f"{leading_spaces}{index}/{max_file_index} "
            f"({leading_spaces_pc}{index_percent:0.1%})"
        )

    def _find_files(
        self,
        path_manager: PathManager,
    ) -> list[tuple[int, str, str]]:
        approved_formats = set(APPROVED_FORMATS_IMAGE + APPROVED_FORMATS_VIDEO)
        full_files_path = path_manager.get_uncensored_folder()

        if not full_files_path.exists():
            msg = f"Path Does Not Exist: {full_files_path}"
            raise TypeError(msg)

        # Recursive glob for all files under the folder
        files = [
            f
            for f in full_files_path.rglob("*")
            if f.is_file() and f.suffix.lower() in approved_formats
        ]

        if not files:
            msg = f"Empty folder: {full_files_path}"
            raise FileNotFoundError(msg)

        indexed_files = [
            (
                index,
                str(file_path),
                "video"
                if file_path.suffix.lower() in APPROVED_FORMATS_VIDEO
                else "image",
            )
            for index, file_path in enumerate(files, start=1)
        ]

        if indexed_files:
            return indexed_files

        # Fallback single file case (not common if directory)
        return [
            (
                1,
                str(full_files_path),
                "video"
                if full_files_path.suffix.lower() in APPROVED_FORMATS_VIDEO
                else "image",
            ),
        ]
