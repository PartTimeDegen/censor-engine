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


class TestBaseFunctions:
    def test_initiate(
        self,
        image_file: tuple[Path, Path],
        args: ParsedArguments,
    ):
        base_dir, _rel_media_path = image_file
        config = args.config
        if config is None:
            msg = "Ignore"
            raise TypeError(msg)

        _ip = EndpointHandler(
            base_dir=base_dir,
            uncensored_base_dir=config.file_handling.folders.uncensored,
            censored_base_dir=config.file_handling.folders.censored,
            using_test_data=args.flags.using_test_data,
            using_shortcut=args.flags.using_shortcut,
        )


class TestMethods:
    def test_handle_shortcut(self, pre_loaded_internal_paths: EndpointHandler):
        ip = pre_loaded_internal_paths

        # Bad Cases
        with pytest.raises(ValueError):
            ip.using_shortcut = False
            ip._handle_shortcut(Path("./thing/file.jpg"))

        with pytest.raises(ValueError):
            ip.using_shortcut = False
            ip._handle_shortcut(Path("thing/file.jpg"))

        with pytest.raises(ValueError):
            ip.using_shortcut = True
            ip._handle_shortcut(Path("uncensored/thing/file.jpg"))

        # Actual Cases
        ip.using_shortcut = False
        assert ip._handle_shortcut(Path("uncensored/thing/file.jpg")) == Path(
            "uncensored/thing/file.jpg"
        )

        ip.using_shortcut = True
        assert ip._handle_shortcut(Path("./thing/file.jpg")) == Path(
            "uncensored/thing/file.jpg"
        )

    def test_handle_test_data(
        self, pre_loaded_internal_paths: EndpointHandler
    ):
        ip = pre_loaded_internal_paths

        ip.using_test_data = False
        assert ip._handle_test_data(Path("uncensored/thing/file.jpg")) == Path(
            "uncensored/thing/file.jpg"
        )

        ip.using_test_data = True
        assert ip._handle_test_data(Path("uncensored/thing/file.jpg")) == Path(
            ".test_data/uncensored/thing/file.jpg"
        )

    def test_handle_flag_cases(
        self, pre_loaded_internal_paths: EndpointHandler
    ):
        ip = pre_loaded_internal_paths

        ip.using_test_data = False
        ip.using_shortcut = False
        assert ip._handle_flag_cases(
            Path("uncensored/thing/file.jpg")
        ) == Path("uncensored/thing/file.jpg")

        ip.using_test_data = False
        ip.using_shortcut = True
        assert ip._handle_flag_cases(Path("./thing/file.jpg")) == Path(
            "uncensored/thing/file.jpg"
        )

        ip.using_test_data = True
        ip.using_shortcut = False
        assert ip._handle_flag_cases(
            Path("uncensored/thing/file.jpg")
        ) == Path(".test_data/uncensored/thing/file.jpg")

        ip.using_test_data = True
        ip.using_shortcut = True
        assert ip._handle_flag_cases(Path("./thing/file.jpg")) == Path(
            ".test_data/uncensored/thing/file.jpg"
        )


class TestProperties:
    def test_properties(self, pre_loaded_internal_paths: EndpointHandler):
        ip = pre_loaded_internal_paths

        base_dir = ip.base_dir
        rel_media = Path("thing/file.jpg")
        uncen = Path("uncensored")
        cen = Path("censored")

        assert ip.relative_uncensored_media_path == uncen / rel_media
        assert (
            ip.absolute_uncensored_media_path == base_dir / uncen / rel_media
        )
        assert ip.media_path == rel_media
        assert ip.relative_censored_media_path == cen / rel_media
        assert ip.absolute_censored_media_path == base_dir / cen / rel_media

    def test_properties_with_missing_media_file(
        self, pre_loaded_internal_paths: EndpointHandler
    ):
        ip = pre_loaded_internal_paths
        ip.raw_uncensored_path = None

        with pytest.raises(ValueError):
            ip.relative_uncensored_media_path
        with pytest.raises(ValueError):
            ip.absolute_uncensored_media_path
        with pytest.raises(ValueError):
            ip.media_path
        with pytest.raises(ValueError):
            ip.relative_censored_media_path
        with pytest.raises(ValueError):
            ip.absolute_censored_media_path

    # TODO: Need to work out the logic for the "handle_flag_cases" with the properties
    def test_properties_with_shortcut(
        self, pre_loaded_internal_paths: EndpointHandler
    ):
        ip = pre_loaded_internal_paths
        ip.using_shortcut = True
        ip.raw_uncensored_path = Path("./thing/file.jpg")

        base_dir = ip.base_dir
        rel_media = Path("thing/file.jpg")
        uncen = Path("uncensored")
        cen = Path("censored")

        assert ip.relative_uncensored_media_path == uncen / rel_media
        assert (
            ip.absolute_uncensored_media_path == base_dir / uncen / rel_media
        )
        assert ip.media_path == rel_media
        assert ip.relative_censored_media_path == cen / rel_media
        assert ip.absolute_censored_media_path == base_dir / cen / rel_media

    def test_properties_with_test_data(
        self, pre_loaded_internal_paths: EndpointHandler
    ):
        ip = pre_loaded_internal_paths
        ip.using_test_data = True
        ip.raw_uncensored_path = Path("uncensored/thing/file.jpg")

        base_dir = ip.base_dir
        test_data = Path(".test_data")
        rel_media = Path("thing/file.jpg")
        uncen = Path("uncensored")
        cen = Path("censored")

        assert (
            ip.relative_uncensored_media_path == test_data / uncen / rel_media
        )
        assert (
            ip.absolute_uncensored_media_path
            == base_dir / test_data / uncen / rel_media
        )
        assert ip.media_path == rel_media
        assert ip.relative_censored_media_path == test_data / cen / rel_media
        assert (
            ip.absolute_censored_media_path
            == base_dir / test_data / cen / rel_media
        )

    def test_properties_with_shortcut_and_test_date(
        self, pre_loaded_internal_paths: EndpointHandler
    ):
        ip = pre_loaded_internal_paths
        ip.using_shortcut = True
        ip.using_test_data = True
        ip.raw_uncensored_path = Path("./thing/file.jpg")

        base_dir = ip.base_dir
        test_data = Path(".test_data")
        rel_media = Path("thing/file.jpg")
        uncen = Path("uncensored")
        cen = Path("censored")

        assert (
            ip.relative_uncensored_media_path == test_data / uncen / rel_media
        )
        assert (
            ip.absolute_uncensored_media_path
            == base_dir / test_data / uncen / rel_media
        )
        assert ip.media_path == rel_media
        assert ip.relative_censored_media_path == test_data / cen / rel_media
        assert (
            ip.absolute_censored_media_path
            == base_dir / test_data / cen / rel_media
        )
