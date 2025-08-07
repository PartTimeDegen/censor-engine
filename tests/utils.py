from dataclasses import dataclass
import inspect
from typing import Any
import cv2
import numpy as np
import yaml
from censor_engine.models.lib_models.detectors import DetectedPartSchema
from tests.input_image import ImageGenerator
from pathlib import Path
from censor_engine import CensorEngine


# Utils
def assert_image(
    output_image: np.ndarray,
    mean_absolute_error: float = 10,
    subfolder: str | None = None,
    batch_tests: bool = False,
    group_name: str | None = None,
    expect_png: bool = False,
    edge_case: bool = False,
):
    # Get caller info to name the dump folder by test name
    stack_trace = 3 if batch_tests else 2
    caller = inspect.stack()[stack_trace]
    caller_file = Path(caller.filename).resolve()
    test_name = caller.function

    if test_name.startswith("test_"):
        test_name = test_name[len("test_") :]

    if group_name:
        if group_name:
            base_path = Path(group_name.removeprefix("test_"))
            if edge_case:
                test_name = base_path / "edge_cases" / test_name
            else:
                test_name = base_path / test_name

    # Base path for image I/O
    dump_path = caller_file.parent / "test_output_data" / test_name
    if subfolder:
        dump_path = dump_path / subfolder
    dump_path.mkdir(parents=True, exist_ok=True)

    # Key files
    ext = "png" if expect_png else "jpg"
    expected_path = dump_path / f"expected.{ext}"
    output_path = dump_path / f"output.{ext}"
    diff_path = dump_path / f"diff.{ext}"
    checkme_flag = dump_path / ".checkme"

    # First time run — no expected baseline
    if not expected_path.exists():
        cv2.imwrite(str(expected_path), output_image)
        cv2.imwrite(str(output_path), output_image)
        checkme_flag.touch()
        raise AssertionError(
            f"No baseline found.\n"
            f"  → Saved expected image to: {expected_path}\n"
            f"  → Review the output at:     {output_path}\n"
            f"  → Then delete {checkme_flag} to accept it.\n"
        )

    # Save artifacts for review
    cv2.imwrite(str(output_path), output_image)

    # Check for `.checkme` marker
    if checkme_flag.exists():
        raise AssertionError(
            f"Baseline approval pending.\n"
            f"  {checkme_flag} exists — review expected/output images at:\n"
            f"    {dump_path}\n"
            f"  Delete the `.checkme` file if you're happy with the result."
        )

    # Load expected image
    expected_image = cv2.imread(str(expected_path), cv2.IMREAD_UNCHANGED)
    if expected_image is None:
        raise AssertionError(f"Failed to load expected image from: {expected_path}")

    # Check shape
    if output_image.shape != expected_image.shape:
        cv2.imwrite(str(output_path), output_image)
        raise AssertionError(
            f"Image shapes differ: {output_image.shape} vs {expected_image.shape}\n"
            f"  → Output saved to: {output_path}"
        )

    # Compare using MAE
    diff = np.abs(output_image.astype(np.int16) - expected_image.astype(np.int16))
    mae = diff.mean()

    if mae <= mean_absolute_error:
        return  # Pass

    # Save artifacts for review
    cv2.imwrite(str(diff_path), diff.astype(np.uint8))

    raise AssertionError(
        f"Image mismatch: MAE={mae:.2f} > {mean_absolute_error}\n"
        f"  → Output image: {output_path}\n"
        f"  → Diff image:   {diff_path}\n"
        f"  → Baseline:     {expected_path}"
    )


def load_config_base_yaml(config: str = "00_default.yml"):
    base_config_path = Path("src") / "censor_engine" / "libs" / "configs" / config
    with open(str(base_config_path), "r") as file:
        return yaml.safe_load(file)


@dataclass
class ImageFixtureData:
    path: Path
    generator: ImageGenerator
    parts: list[DetectedPartSchema]

    def update_parts(self, list_parts_enabled: list[str] | str) -> None:
        if isinstance(list_parts_enabled, str):
            list_parts_enabled = [list_parts_enabled]

        self.parts = [part for part in self.parts if part.label in list_parts_enabled]
        if not self.parts:
            raise ValueError(f"Missing parts: {list_parts_enabled}")


def run_image_test(
    dummy_image_data: ImageFixtureData,
    config: str | dict[str, Any],
    subfolder: str | None = None,
    batch_tests: bool = False,
    group_name: str | None = None,
    expect_png: bool = False,
    edge_case: bool = False,
    mean_absolute_error: float = 6,
):
    if isinstance(config, str):
        config = load_config_base_yaml(config)

    output = CensorEngine(
        base_folder=dummy_image_data.path.parent,
        config_data=config,
        _test_mode=True,
        _test_detection_output=dummy_image_data.generator.return_detected_parts(),
    ).start()[0]

    assert_image(
        output,
        subfolder=subfolder,
        mean_absolute_error=mean_absolute_error,
        batch_tests=batch_tests,
        group_name=group_name,
        expect_png=expect_png,
        edge_case=edge_case,
    )
