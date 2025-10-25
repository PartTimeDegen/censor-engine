import inspect
import shutil
from pathlib import Path

import cv2
import numpy as np


def write_video(path: Path, frames: np.ndarray, fps: int = 30) -> None:
    """
    Save a sequence of frames (H,W,C) into a video file.
    frames: ndarray shape (N, H, W, C)
    """
    h, w, c = frames.shape[1:]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame.astype(np.uint8))
    out.release()


def read_video(path: Path) -> np.ndarray:
    """Load all frames from a video file into a numpy array (N,H,W,C)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    return np.stack(frames, axis=0)


def assert_video(
    output_path: Path,
    mean_absolute_error: float = 10,
    subfolder: str | None = None,
    batch_tests: bool = False,
    group_name: str | None = None,
    edge_case: bool = False,
) -> None:
    # Get caller info to name the dump folder by test name
    stack_trace = 3 if batch_tests else 2
    caller = inspect.stack()[stack_trace]
    caller_file = (
        Path(caller.filename).resolve().relative_to(Path("tests").resolve())
    )
    test_name = caller.function.removeprefix("test_")

    if group_name:
        base_path = Path(group_name.removeprefix("test_"))
        test_name = base_path / ("edge_cases" if edge_case else "") / test_name

    dump_path = Path("tests") / "00_test_data" / caller_file.parent / test_name
    if subfolder:
        dump_path = dump_path / subfolder
    dump_path.mkdir(parents=True, exist_ok=True)

    expected_path = dump_path / "expected.mp4"
    review_output_path = dump_path / "output.mp4"
    diff_path = dump_path / "diff.mp4"
    checkme_flag = dump_path / ".checkme"

    # Copy the produced video into the dump folder
    shutil.copy(output_path, review_output_path)

    # First-time run
    if not expected_path.exists():
        shutil.copy(output_path, expected_path)
        checkme_flag.touch()
        raise AssertionError(
            f"No baseline found.\n"
            f"  → Saved expected video to: {expected_path}\n"
            f"  → Review the output at:     {review_output_path}\n"
            f"  → Then delete {checkme_flag} to accept it.\n"
        )

    # Still waiting for approval?
    if checkme_flag.exists():
        raise AssertionError(
            f"Baseline approval pending.\n"
            f"  {checkme_flag} exists — review videos at:\n"
            f"    {dump_path}\n"
            f"  Delete the `.checkme` file if you're happy with the result."
        )

    # Load videos into arrays
    expected_frames = read_video(expected_path)
    output_frames = read_video(review_output_path)
    if expected_frames is None or output_frames is None:
        raise AssertionError("Failed to load one of the video files.")

    if expected_frames.shape != output_frames.shape:
        raise AssertionError(
            f"Video shapes differ: {output_frames.shape} vs {expected_frames.shape}\n"
            f"  → Output saved to: {review_output_path}"
        )

    # Compare with MAE
    diff = np.abs(
        output_frames.astype(np.int16) - expected_frames.astype(np.int16)
    )
    mae = diff.mean()

    if mae > mean_absolute_error:
        write_video(diff_path, diff.clip(0, 255).astype(np.uint8))
        raise AssertionError(
            f"Video mismatch: MAE={mae:.2f} > {mean_absolute_error}\n"
            f"  → Output video: {review_output_path}\n"
            f"  → Diff video:   {diff_path}\n"
            f"  → Baseline:     {expected_path}"
        )
