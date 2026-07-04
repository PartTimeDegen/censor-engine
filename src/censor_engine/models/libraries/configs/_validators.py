from enum import Enum
from typing import Any, TypeVar

from censor_engine.models.core.tools.debugger.enums import DebugLevel
from censor_engine.models.enums import MergeMethod, PartState
from censor_engine.models.libraries.configs.settings.schemas import Margins
from censor_engine.structs.censors import Censor


# --- Normalisers --- #
def normalise_censors(
    censor_names: str | dict[str, Any] | list[str] | list[dict[str, Any]],
) -> list[Censor]:

    list_of_censors = (
        [censor_names]
        if isinstance(censor_names, (str, dict))
        else censor_names
    )

    processed: list[Censor] = []
    for item in list_of_censors:
        match item:
            case str():
                processed.append(Censor(item))
            case dict():
                processed.append(Censor(**item))
            case _:
                msg = f"Invalid censor input: {type(item)}"
                raise TypeError(msg)

    return processed


def normalise_margins(
    margin_data: float | dict[str, float | int],
) -> Margins:
    if isinstance(margin_data, (int, float)):
        return Margins(height=margin_data, width=margin_data)

    if isinstance(margin_data, dict):
        return Margins(**margin_data)

    msg = "Wrong Type for Margin Data"
    raise TypeError(msg)


# --- Convertors --- #
E = TypeVar("E", bound=Enum)


def _generic_convert_to_enum[E: Enum](
    input_str: str | type[E],
    enum: type[E],
    enum_name: str,
) -> E:
    if isinstance(input_str, enum):
        return input_str

    if not isinstance(input_str, str):
        msg = f"Invalid {enum_name} value: {input_str}"
        raise TypeError(msg)
    try:
        return getattr(enum, input_str.upper())

    except AttributeError as e:
        msg = f"Invalid {enum_name} value: {input_str}"
        raise AttributeError(msg) from e


def convert_merge_method(merge_method: str | MergeMethod) -> MergeMethod:
    return _generic_convert_to_enum(merge_method, MergeMethod, "MergeMethod")  # type: ignore


def convert_debug_level(debug_level: str | DebugLevel) -> DebugLevel:
    return _generic_convert_to_enum(debug_level, DebugLevel, "DebugLevel")  # type: ignore


def convert_state(state: str | PartState) -> PartState:
    return _generic_convert_to_enum(state, PartState, "PartState")  # type: ignore


# Fixers
# TODO: Need to do the Registry first
# def fix_enabled_detections(input_data: str | list[str]) -> list[str]:
#     if isinstance(input_data, list):
#         return input_data

#     if input_data != "all":
