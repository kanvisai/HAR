"""Load poses_full.npy and the geometry of the 8-point skeleton."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Fixed order in the .npy (YOLO pose, upper body).
NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)
SHOULDER_L, SHOULDER_R = 0, 1
ELBOW_L, ELBOW_R = 2, 3
WRIST_L, WRIST_R = 4, 5
HIP_L, HIP_R = 6, 7

BONES = (
    (SHOULDER_L, SHOULDER_R),
    (SHOULDER_L, ELBOW_L),
    (ELBOW_L, WRIST_L),
    (SHOULDER_R, ELBOW_R),
    (ELBOW_R, WRIST_R),
    (SHOULDER_L, HIP_L),
    (SHOULDER_R, HIP_R),
    (HIP_L, HIP_R),
)

ARM_SIDE = {
    "left": (SHOULDER_L, ELBOW_L, WRIST_L, HIP_L),
    "right": (SHOULDER_R, ELBOW_R, WRIST_R, HIP_R),
}

EXAMPLE_POSES = Path(
    "/home/ignacio/Escritorio/Company/HAR_en/videos/6/"
    "robos_3_clips_clip61_000000_000020_6/user_15361/poses_full.npy"
)
EXAMPLE_VIDEO = Path(
    "/home/ignacio/Escritorio/Company/HAR_en/videos/6/"
    "robos_3_clips_clip61_000000_000020_6/clip.mp4"
)


def load_poses(path: Path) -> np.ndarray:
    """Return (frames, 8, 2) in normalized coordinates, y pointing down."""
    poses = np.load(path).astype(np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (8, 2):
        raise ValueError(
            f"{path} has shape {poses.shape}. Expected (frames, 8, 2)."
        )
    return poses


def fps_and_video(poses_path: Path, video: Path | None = None) -> tuple[float, Path | None]:
    """Read fps from the clip meta.json and locate clip.mp4 when it is not given."""
    clip_dir = poses_path.parent.parent if poses_path.parent.name.startswith("user_") else poses_path.parent
    meta_path = clip_dir / "meta.json"
    fps = 12.5
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fps = float(meta.get("fps", fps))
        if video is None:
            name = meta.get("clip_video", "clip.mp4")
            candidate = clip_dir / name
            if candidate.is_file():
                video = candidate
    if video is None:
        candidate = clip_dir / "clip.mp4"
        if candidate.is_file():
            video = candidate
    return fps, video


def smooth(poses: np.ndarray, window: int = 5) -> np.ndarray:
    """Temporal median. Does not fill gaps: a NaN inside the window stays NaN."""
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and >= 3")
    half = window // 2
    padded = np.pad(poses, ((half, half), (0, 0), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    return np.median(windows, axis=-1)


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:05.2f}"
