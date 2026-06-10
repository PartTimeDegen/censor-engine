from natsort import natsorted

from censor_engine.models.structs import IndexedFile, Mixin
from censor_engine.paths import PathManager

APPROVED_FORMATS_IMAGE = [".jpg", ".jpeg", ".png", ".webp"]
APPROVED_FORMATS_VIDEO = [".mp4", ".webm", ".mov"]


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
    ) -> list[IndexedFile]:
        approved_formats = set(APPROVED_FORMATS_IMAGE + APPROVED_FORMATS_VIDEO)
        full_files_path = path_manager.get_uncensored_folder()

        if path_manager.test_mode:
            example_image = full_files_path / "config_example.jpg"
            return [IndexedFile(1, str(example_image), "preview")]

        if not full_files_path.exists():
            full_files_path.mkdir(exist_ok=True)

        # Singular File Fix
        if full_files_path.is_file():
            if full_files_path.suffix.lower() not in approved_formats:
                msg = "File Doesn't have an approved format"
                raise TypeError(msg)
            return [
                IndexedFile(
                    1,
                    str(full_files_path),
                    "video"
                    if full_files_path.suffix.lower() in APPROVED_FORMATS_VIDEO
                    else "image",
                )
            ]

        # Recursive glob for all files under the folder
        files = natsorted(
            [
                f
                for f in full_files_path.rglob("*")
                if f.is_file() and f.suffix.lower() in approved_formats
            ],
            key=lambda p: p.name,
        )

        if not files:
            msg = f"Empty folder: {full_files_path}"
            raise FileNotFoundError(msg)

        indexed_files = [
            IndexedFile(
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
            IndexedFile(
                1,
                str(full_files_path),
                "video"
                if full_files_path.suffix.lower() in APPROVED_FORMATS_VIDEO
                else "image",
            ),
        ]
