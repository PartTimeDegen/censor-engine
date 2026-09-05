from pathlib import Path

import pytest

from censor_engine.models.core.cli.structs import (
    EngineFlags,
    ParsedArguments,
)
from censor_engine.models.core.path_manager._endpoint_handler import (
    EndpointHandler,
)
from censor_engine.models.core.path_manager.path_manager import (
    PathManager,
)
from censor_engine.models.core.path_manager.schemas import (
    FileType,
    IndexedFile,
)
from censor_engine.models.core.tools.debugger.enums import DebugLevel
from censor_engine.models.libraries.configs.config import Config


@pytest.fixture
def image_file(tmp_path) -> tuple[Path, Path]:
    rel_path = Path("uncensored") / "thing" / "file.jpg"
    media = Path(tmp_path) / rel_path

    # Make Contents
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"hello world")
    return (tmp_path, rel_path)


@pytest.fixture
def video_file(tmp_path) -> tuple[Path, Path]:
    rel_path = Path("uncensored") / "thing" / "file.webm"
    media = Path(tmp_path) / rel_path

    # Make Contents
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"hello world")
    return (tmp_path, rel_path)


@pytest.fixture
def args() -> ParsedArguments:
    return ParsedArguments(
        flags=EngineFlags(),
        uncensored_location_override=None,
        debug_level=DebugLevel.NONE,
        config=Config.from_dict({}),
    )


# Full Fixtures
@pytest.fixture
def pre_loaded_internal_paths(
    image_file: tuple[Path, Path], args: ParsedArguments
) -> EndpointHandler:
    base_dir, rel_media_path = image_file
    config = args.config
    if config is None:
        msg = "Ignore"
        raise TypeError(msg)
    return EndpointHandler(
        base_dir=base_dir,
        uncensored_base_dir=config.file_handling.folders.uncensored,
        censored_base_dir=config.file_handling.folders.censored,
        using_test_data=args.flags.using_test_data,
        using_shortcut=args.flags.using_shortcut,
        raw_uncensored_path=rel_media_path,
    )


@pytest.fixture
def pre_loaded_path_manager(
    image_file: tuple[Path, Path], args: ParsedArguments
) -> PathManager:
    base_dir, rel_media_path = image_file
    pm = PathManager(base_dir, args)
    pm.load_media_path_into_manager(relative_media_path=rel_media_path)
    return pm


class TestPathManager:
    def test_initiate(
        self,
        image_file: tuple[Path, Path],
        args: ParsedArguments,
    ):
        base_dir, _rel_media_path = image_file
        PathManager(base_dir, args)

    class TestMethods:
        class TestLoadMediaPath:
            def test_baseline(
                self,
                image_file: tuple[Path, Path],
                args: ParsedArguments,
            ):
                base_dir, rel_media_path = image_file
                pm = PathManager(base_dir, args)

                censored_path = pm.load_media_path_into_manager(
                    relative_media_path=rel_media_path
                )

                # Assertion Prep
                rel_media_parts = rel_media_path.parts
                new_path = Path("censored", *rel_media_parts[1:])

                assert censored_path == base_dir / new_path
                assert censored_path.exists()

        class TestGetFiles:
            def test_baseline(self, pre_loaded_path_manager: PathManager):
                pm = pre_loaded_path_manager
                files = pm.get_files()

                assert isinstance(files, list)
                assert isinstance(files[0], IndexedFile)

                assert files[0].file_type == FileType.IMAGE
                assert files[0].index == 1
                assert files[0].path == pm.paths.absolute_uncensored_media_path

    class TestProperties:
        class TestFolderForOutputDisplay:
            def test_baseline(self, pre_loaded_path_manager: PathManager):
                pm = pre_loaded_path_manager

                # Only the File Name
                assert (
                    pre_loaded_path_manager.folder_for_output_display
                    == "file.jpg"
                )

                # Relative File Name
                pm._flags.display_full_output = True
                assert pm.folder_for_output_display == str(
                    pm.paths.relative_censored_media_path
                )
