from pathlib import Path

from tests.helpers.test_data_handler import handle_test_data


def run_effect_tester(file_path, effect_context, effect_class, param, value):
    args = {param: value} if param is not None else {}

    effect = effect_class()

    effect_image = effect.generate_effect(effect_context, **args)
    output = effect.apply_effect_to_image(effect_context, effect_image)

    # cv2.imwrite(f"_{param}_{value}.png", output)
    title = f"{param}-{value}" if param is not None else "baseline"
    handle_test_data(
        test_name=str(Path(effect_class.__name__) / title),
        image=output,
        file_path=file_path,
    )
