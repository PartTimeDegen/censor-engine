from dataclasses import dataclass, field
from pathlib import Path

from natsort import natsorted

from ._constants import CONFIG_PREVIEW
from ._structs import ApprovedFormats
from .schemas import FileType, IndexedFile


@dataclass(slots=True)
class FileExplorer:
    root_path: Path
    preview_mode: bool
    approved_formats: ApprovedFormats = field(default_factory=ApprovedFormats)

    def _make_file_indexed(
        self,
        file_path: Path,
        index: int = 1,
        file_type: FileType | None = None,
    ) -> IndexedFile:
        # Get File Type
        return IndexedFile(
            index=index,
            path=file_path,
            file_type=self.approved_formats.get_format_type(file_path)
            if file_type is None
            else file_type,
        )

    def _index_single_file(self, file_path: Path) -> list[IndexedFile]:
        if not self.approved_formats.check_is_approved(file_path):
            msg = f"File Doesn't have an approved format {file_path.suffix}"
            raise TypeError(msg)

        return [self._make_file_indexed(file_path)]

    def _index_multiple_files(self, directory: Path) -> list[IndexedFile]:
        # Get the list of files
        files = natsorted(
            (
                found
                for found in directory.rglob("*")
                if self.approved_formats.check_is_approved(found)
            ),
            key=lambda p: p.name,
        )

        # If No Files
        if not files:
            msg = f"Empty folder: {directory}"
            raise FileNotFoundError(msg)

        return [
            self._make_file_indexed(found, index=index)
            for index, found in enumerate(files, start=1)
        ]

    def find_files(self) -> list[IndexedFile]:
        # Preview Mode
        if self.preview_mode:
            file_path = self.root_path / CONFIG_PREVIEW
            return [
                self._make_file_indexed(file_path, file_type=FileType.PREVIEW)
            ]

        # Check if root_path is a File
        if self.root_path.is_file():
            return self._index_single_file(self.root_path)

        return self._index_multiple_files(self.root_path)
