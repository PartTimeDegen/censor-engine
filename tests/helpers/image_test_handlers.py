from pathlib import Path

from censor_engine.models.libraries.masks._base_mask import Mask
from censor_engine.models.libraries.masks.schemas import MaskContext
from tests.helpers.test_data_handler import handle_test_data


def general_image_library_test(
    mask_obj: Mask,
    mask_context: MaskContext,
    file_path: Path,
    method_used: str,
    prefix: Path | str | None = None,
    include_shape_name: bool = True,
) -> None:
    if isinstance(prefix, Path):
        prefix = str(prefix)
    # Run Generate Test
    init_obj = mask_obj()  # type: ingore
    method = getattr(init_obj, method_used)
    output = method(mask_context)  # type: ignore

    # Handle Naming
    name = mask_obj.__name__  # type: ignore
    if prefix is not None:
        if include_shape_name:
            name = str(Path(prefix) / name)
        else:
            name = str(Path(prefix))

    # Run the Equality Checker
    handle_test_data(name, output, file_path)
