import hashlib
from pathlib import Path

import pytest

from censor_engine import models
from censor_engine.models.core.data_caching._common_paths import (
    META_FILE,
)
from censor_engine.models.core.data_caching._common_schemas import (
    MetaData,
)
from censor_engine.models.core.data_caching._utils import (
    check_cache_hash_matches_file,
    create_cache_folder,
    create_meta_file,
    delete_cache_folder_contents,
    get_hash,
)


class TestGetHash:
    def test_returns_correct_sha256(self, tmp_path):
        media = tmp_path / "file.txt"
        content = b"hello world"
        media.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()

        assert get_hash(media) == expected

    def test_empty_file(self, tmp_path):
        media = tmp_path / "empty.txt"
        media.write_bytes(b"")

        expected = hashlib.sha256(b"").hexdigest()

        assert get_hash(media) == expected

    def test_large_file_multiple_chunks(self, tmp_path):
        media = tmp_path / "large.bin"

        # Larger than the 64 KB chunk size used by get_hash()
        content = b"x" * 100_000
        media.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()

        assert get_hash(media) == expected


class TestCreateCacheFolder:
    def test_creates_new_directory(self, tmp_path):
        relative_path = Path("some_folder/image.png")
        cache_dir = tmp_path / ".cache" / relative_path

        result = create_cache_folder(cache_dir)

        assert result == tmp_path / ".cache" / "some_folder/image.png"
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_removes_existing_contents(self, tmp_path):
        relative_path = Path("some_folder/image.png")
        cache_dir = tmp_path / ".cache" / relative_path
        cache_dir.mkdir(parents=True)

        old_file = cache_dir / "old.txt"
        old_file.write_text("stale data")

        assert old_file.exists()

        create_cache_folder(cache_dir)

        assert cache_dir.exists()
        assert cache_dir.is_dir()


class TestDeleteCacheFolderContents:
    def test_deletes_directory_and_all_contents(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache"
        cache_path.mkdir()

        (cache_path / "file.txt").write_text("test")
        (cache_path / "subdir").mkdir()
        (cache_path / "subdir" / "nested.txt").write_text("nested")

        delete_cache_folder_contents(cache_path)

        assert not cache_path.exists()

    def test_raises_for_non_existent_directory(self) -> None:
        with pytest.raises(FileNotFoundError):
            delete_cache_folder_contents(Path("/does/not/exist"))


class TestCreateMetaFile:
    def test_creates_meta_file(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        meta_file = create_meta_file(cache_dir, "abc123")

        assert meta_file.exists()
        assert meta_file.name == META_FILE

    def test_writes_correct_json(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        hash_value = "abc123"

        meta_file = create_meta_file(cache_dir, hash_value)

        meta = MetaData.model_validate_json(meta_file.read_text())

        assert meta.hash_data == hash_value


class TestCheckCacheHashMatchesFile:
    def test_returns_false_when_meta_file_missing(self, tmp_path):
        media = tmp_path / "media.jpg"
        media.write_bytes(b"content")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        assert check_cache_hash_matches_file(media, cache_dir) is False

    def test_returns_true_when_hash_matches(self, tmp_path):
        media = tmp_path / "media.jpg"
        media.write_bytes(b"content")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        create_meta_file(cache_dir, get_hash(media))

        assert check_cache_hash_matches_file(cache_dir, media) is True

    def test_returns_false_when_hash_does_not_match(self, tmp_path):
        media = tmp_path / "media.jpg"
        media.write_bytes(b"content")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        create_meta_file(cache_dir, "incorrect_hash")

        assert check_cache_hash_matches_file(media, cache_dir) is False
