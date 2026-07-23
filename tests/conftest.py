import os
from pathlib import Path

ignore_paths = ["__init__", "__pycache__"]
fixtures = [
    str(path)[:-3].replace(os.sep, ".")
    for path in list(Path("tests/fixtures").glob("*"))
    if path.stem not in ignore_paths and not path.stem.startswith("_")
]
print(*fixtures, sep="\n")


pytest_plugins = fixtures
