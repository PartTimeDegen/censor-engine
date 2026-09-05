from pathlib import Path

from tests.helpers.test_data_handler import handle_test_data


def run_effect_tester(
    file_path,
    effect_context,
    effect_class,
    param,
    value,
    extra_params=None,
    secondary=None,
):
    args = {param: value} if param is not None else {}
    extra_args = extra_params if extra_params is not None else {}
    effect = effect_class()

    effect_image = effect.generate_effect(effect_context, **args, **extra_args)
    output = effect.apply_effect_to_image(effect_context, effect_image)

    # cv2.imwrite(f"_{param}_{value}.png", output)
    title = f"{param}-{value}" if param is not None else "baseline"
    base_path = Path(effect_class.__name__)
    if secondary is not None:
        base_path = base_path / secondary

    base_path = base_path / title
    handle_test_data(
        test_name=str(base_path),
        image=output,
        file_path=file_path,
    )
