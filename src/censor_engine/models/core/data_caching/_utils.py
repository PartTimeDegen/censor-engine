import hashlib
import shutil
from pathlib import Path

from ._common_paths import CACHE_FOLDER, META_FILE
from ._common_schemas import MetaData


def get_hash(media_path: Path) -> str:
    """
    This function is used to get the hash of an image/video in order to confirm
    it's the same as the one it's using the cache for. This is used to avoid
    situations where the name is the same but the thing itself is different.

    :param Path media_path: Path to the media
    :return str: Hash value
    """
    # Get SHA
    sha = hashlib.sha256()
    buffer_size = 65_536  # 64 KB chunks

    # Read File
    with media_path.open(mode="rb") as f:
        while chunk := f.read(buffer_size):
            sha.update(chunk)

    # Reduce to Hash
    return sha.hexdigest()


def check_cache_hash_matches_file(cache_path: Path, media_path: Path) -> bool:
    # Get the Meta File
    meta_file = cache_path / META_FILE
    if not meta_file.exists():
        return False

    # Read Meta File
    with meta_file.open() as f:
        meta_data = f.read()

    # Check Hash from the file
    meta_object = MetaData.model_validate_json(meta_data)
    found_media_hash = meta_object.hash_data

    # Confirm Hashes Match
    return found_media_hash == get_hash(media_path)


def get_cache_path(base_dir: Path, relative_path: Path) -> Path:
    return base_dir / CACHE_FOLDER / relative_path


def delete_cache_folder_contents(cache_path: Path) -> None:
    shutil.rmtree(str(cache_path))


def create_cache_folder(cache_path: Path) -> Path:
    # Make Folder
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def create_meta_file(cache_path: Path, hash_data: str) -> Path:
    # Get Meta File
    meta_file = cache_path / META_FILE

    # Write File
    meta_entry = MetaData(hash_data=hash_data)
    with meta_file.open("w") as f:
        f.write(meta_entry.model_dump_json())

    # Return Path
    return meta_file


# TODO: blah
"""
The current plan is to have the general functions be functions, then have the
core stuff for the image cache and the video cache.

I might make them their own folder so it's not messy.

I also need to go through the hell of renaming the private functions to private
names.

Writing the individual ones, I really need to write docs for them, also I need
to isolate the stuff to an ORM for video. I want to make the code modular
"""
