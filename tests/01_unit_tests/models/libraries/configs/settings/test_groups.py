import pytest
from pydantic import ValidationError

from censor_engine.models.libraries.configs.settings._groups import (
    GroupSettings,
)


def test_group_settings_not_list_of_strings():
    with pytest.raises(ValidationError):
        GroupSettings(persistance=["placeholder"])  # type: ignore
