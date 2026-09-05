from pathlib import Path

import pytest

from censor_engine.models.core.path_manager._constants import CONFIG_PREVIEW
from censor_engine.models.core.path_manager._file_explorer import FileExplorer
from censor_engine.models.core.path_manager.schemas import (
    FileType,
    IndexedFile,
)


@pytest.fixture
def image_file_file_explorer(tmp_path) -> FileExplorer:
    file_path = tmp_path / "file.png"
    file_path.touch()
    return FileExplorer(tmp_path, preview_mode=False)


@pytest.fixture
def video_file_file_explorer(tmp_path) -> FileExplorer:
    file_path = tmp_path / "file.mp4"
    file_path.touch()
    return FileExplorer(tmp_path, preview_mode=False)


@pytest.fixture
def bad_file_file_explorer(tmp_path) -> FileExplorer:
    file_path = tmp_path / "file.aaaaaaaa"
    file_path.touch()
    return FileExplorer(tmp_path, preview_mode=False)


@pytest.fixture
def multiple_files_file_explorer(tmp_path) -> FileExplorer:
    (tmp_path / "a.png").touch()
    (tmp_path / "b.jpg").touch()
    (tmp_path / "10.jpeg").touch()
    (tmp_path / "thing.mp4").touch()
    (tmp_path / "barred.aaaaaaa").touch()
    return FileExplorer(tmp_path, preview_mode=False)


@pytest.fixture
def empty_folder_file_explorer(tmp_path) -> FileExplorer:
    return FileExplorer(tmp_path, preview_mode=False)


@pytest.fixture
def empty_dir(tmp_path):
    return tmp_path


class TestFileExplorer:
    class TestMethods:
        class TestMakeFileIndexed:
            def test_baseline_image(
                self, image_file_file_explorer: FileExplorer
            ):
                fe = image_file_file_explorer

                indexed_file = fe._make_file_indexed(Path("placeholder.png"))

                assert isinstance(indexed_file, IndexedFile)
                assert indexed_file.path == Path("placeholder.png")
                assert indexed_file.file_type == FileType.IMAGE
                assert indexed_file.index == 1

            def test_baseline_video(
                self, image_file_file_explorer: FileExplorer
            ):
                fe = image_file_file_explorer

                indexed_file = fe._make_file_indexed(
                    Path("placeholder.mp4"), index=5
                )

                assert isinstance(indexed_file, IndexedFile)
                assert indexed_file.path == Path("placeholder.mp4")
                assert indexed_file.file_type == FileType.VIDEO
                assert indexed_file.index == 5

            def test_file_type_override(
                self, image_file_file_explorer: FileExplorer
            ):
                fe = image_file_file_explorer

                indexed_file = fe._make_file_indexed(
                    Path("placeholder.png"), file_type=FileType.PREVIEW
                )

                assert indexed_file.file_type == FileType.PREVIEW

        class TestIndexSingleFile:
            def test_baseline(self, image_file_file_explorer: FileExplorer):
                fe = image_file_file_explorer

                indexed_file = fe._index_single_file(Path("file.png"))

                assert isinstance(indexed_file, list)
                assert isinstance(indexed_file[0], IndexedFile)
                assert indexed_file[0].path == Path("file.png")
                assert indexed_file[0].index == 1
                assert indexed_file[0].file_type == FileType.IMAGE

            def test_not_approved(
                self, image_file_file_explorer: FileExplorer
            ):
                fe = image_file_file_explorer

                with pytest.raises(TypeError):
                    fe._index_single_file(Path("file.aaaaaa"))

        class TestIndexMultipleFiles:
            def test_baseline(
                self, multiple_files_file_explorer: FileExplorer
            ):
                fe = multiple_files_file_explorer
                rp = fe.root_path

                files = fe._index_multiple_files(rp)

                assert isinstance(files, list)
                assert isinstance(files[0], IndexedFile)
                assert len(files) == 4  # Ignore Bad File

                assert files[0].path == rp / Path("10.jpeg")
                assert files[0].index == 1
                assert files[0].file_type == FileType.IMAGE

            def test_empty(self, empty_folder_file_explorer: FileExplorer):
                fe = empty_folder_file_explorer
                rp = fe.root_path
                with pytest.raises(FileNotFoundError):
                    fe._index_multiple_files(rp)

        class TestFindFiles:
            def test_baseline(
                self, multiple_files_file_explorer: FileExplorer
            ):
                fe = multiple_files_file_explorer
                rp = fe.root_path

                files = fe.find_files()

                assert isinstance(files, list)
                assert isinstance(files[0], IndexedFile)
                assert len(files) == 4  # Ignore Bad File

                assert files[0].path == rp / Path("10.jpeg")
                assert files[0].index == 1
                assert files[0].file_type == FileType.IMAGE
                assert files[3].file_type == FileType.VIDEO

            def test_config_preview(
                self, multiple_files_file_explorer: FileExplorer
            ):
                fe = multiple_files_file_explorer
                rp = fe.root_path

                fe.preview_mode = True
                files = fe.find_files()

                assert isinstance(files, list)
                assert isinstance(files[0], IndexedFile)
                assert len(files) == 1

                assert files[0].path == rp / Path(CONFIG_PREVIEW)
                assert files[0].index == 1
                assert files[0].file_type == FileType.PREVIEW

            def test_single_file_image(
                self, image_file_file_explorer: FileExplorer
            ):
                fe = image_file_file_explorer
                rp = fe.root_path

                files = fe.find_files()

                assert isinstance(files, list)
                assert isinstance(files[0], IndexedFile)
                assert len(files) == 1  # Ignore Bad File

                assert files[0].path == rp / Path("file.png")
                assert files[0].index == 1
                assert files[0].file_type == FileType.IMAGE

            def test_single_file_video(
                self, video_file_file_explorer: FileExplorer
            ):
                fe = video_file_file_explorer
                rp = fe.root_path

                files = fe.find_files()

                assert isinstance(files, list)
                assert isinstance(files[0], IndexedFile)
                assert len(files) == 1  # Ignore Bad File

                assert files[0].path == rp / Path("file.mp4")
                assert files[0].index == 1
                assert files[0].file_type == FileType.VIDEO

            def test_multiple_files(
                self, multiple_files_file_explorer: FileExplorer
            ):
                fe = multiple_files_file_explorer
                rp = fe.root_path

                files = fe.find_files()

                assert isinstance(files, list)
                assert isinstance(files[0], IndexedFile)
                assert len(files) == 4  # Ignore Bad File

                assert files[0].path == rp / Path("10.jpeg")
                assert files[0].index == 1
                assert files[0].file_type == FileType.IMAGE
                assert files[3].file_type == FileType.VIDEO

            def test_empty_folder(
                self, empty_folder_file_explorer: FileExplorer
            ):
                fe = empty_folder_file_explorer

                with pytest.raises(FileNotFoundError):
                    fe.find_files()
