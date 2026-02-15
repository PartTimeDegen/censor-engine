import shutil
from pathlib import Path

SRC_ROOT = Path("tests")
DST_ROOT = Path("res")

for expected in SRC_ROOT.rglob("expected.jpg"):
    # Folder containing expected.jpg
    folder = expected.parent

    # Relative path from tests/
    rel_path = folder.relative_to(SRC_ROOT)

    # New filename = folder name + .jpg
    new_name = f"{folder.name}.jpg"

    # Destination path
    dst_path = DST_ROOT / rel_path.parent / new_name

    # Create destination directories if needed
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy file
    shutil.copy2(expected, dst_path)

    print(f"Copied: {expected} -> {dst_path}")  # noqa: T201
