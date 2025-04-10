import os
import pytest

DEBUG = True


@pytest.fixture(scope="function")
def root_path():
    file_path = os.path.dirname(__file__)  # noqa: F821
    list_path = file_path.split(os.sep)
    list_path = list_path[: list_path.index("tests")]
    yield os.sep.join([*list_path, "src"])
