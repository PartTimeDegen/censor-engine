from pathlib import Path

import pytest

from censor_engine.models.core.path_manager.schemas import (
    FileType,
    IndexedFile,
)

inputs = [
    (1, 10),
    (5, 10),
    (10, 10),
    (1, 100),
    (25, 100),
    (100, 100),
]
outputs = [
    " 1/10 ( 10.0%)",
    " 5/10 ( 50.0%)",
    "10/10 (100.0%)",
    "  1/100 (  1.0%)",
    " 25/100 ( 25.0%)",
    "100/100 (100.0%)",
]

both = [
    (index, max_index, expected)
    for (index, max_index), expected in zip(inputs, outputs, strict=False)
]


@pytest.mark.parametrize(("index", "max_index", "expected"), both)
def test_get_index(index: int, max_index: int, expected: str):
    indexed_file = IndexedFile(
        index=index,
        path=Path("dummy.png"),
        file_type=FileType.IMAGE,
    )

    assert indexed_file.get_index(max_index) == expected
