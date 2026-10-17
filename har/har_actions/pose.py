"""Load poses_full.npy and the geometry of the 8-point skeleton."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

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

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = Path("/home/ignacio/Escritorio/Company/videos")


def load_poses(path: Path) -> np.ndarray:
    """Return (frames, 8, 2) in normalized coordinates, y pointing down."""
    poses = np.load(path).astype(np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (8, 2):
        raise ValueError(f"{path} has shape {poses.shape}. Expected (frames, 8, 2).")
    return poses


def clip_directory(poses_path: Path) -> Path:
    """Map videos/<cat>/<clip>/user_N/poses_full.npy to the clip folder."""
    parent = poses_path.parent
    if parent.name.startswith("user_"):
        clip = parent.parent
        if (clip / "meta.json").is_file() or (clip / "clip.mp4").is_file():
            return clip
    clip_name = parent.name
    set_dir = parent.parent
    if set_dir.name.endswith("_npy"):
        source = set_dir.parent / set_dir.name[: -len("_npy")] / clip_name
        if source.is_dir():
            return source
    return parent


def fps_and_video(poses_path: Path, video: Path | None = None) -> tuple[float, Path | None]:
    """Read fps from meta.json and locate clip.mp4 beside the source clip."""
    clip_dir = clip_directory(poses_path)
    meta_path = clip_dir / "meta.json"
    fps = 12.5
    meta_has_fps = False
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "fps" in meta:
            fps = float(meta["fps"])
            meta_has_fps = True
        if video is None:
            name = meta.get("clip_video") or meta.get("clip_filename") or "clip.mp4"
            candidate = clip_dir / name
            if candidate.is_file():
                video = candidate
    if video is None:
        for name in ("clip.mp4", "clip_buffer.mp4"):
            candidate = clip_dir / name
            if candidate.is_file():
                video = candidate
                break
    if video is None:
        sibling = clip_dir.parent / f"{clip_dir.name}.mp4"
        if sibling.is_file():
            video = sibling
    if video is not None and video.is_file() and not meta_has_fps:
        import cv2

        capture = cv2.VideoCapture(str(video))
        read_fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if read_fps > 1:
            fps = read_fps
    return fps, video


def frame_aspect(video: Path | None) -> float:
    """Width divided by height, so a horizontal arm is not measured short."""
    if video is not None and video.is_file():
        import cv2

        capture = cv2.VideoCapture(str(video))
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if width > 0 and height > 0:
            return width / height
    return 16 / 9


def smooth(poses: np.ndarray, window: int = 3) -> np.ndarray:
    """Temporal median. A missing point stays missing; it is not filled in."""
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and >= 3")
    half = window // 2
    padded = np.pad(poses, ((half, half), (0, 0), (0, 0)), mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(windows, axis=-1)


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:05.2f}"


def seen_pose(pose: np.ndarray) -> np.ndarray:
    """Copy of the pose. Points that were not seen are already NaN and are not invented."""
    return np.array(pose, dtype=np.float64, copy=True)


EXAMPLE_POSES = sorted(VIDEOS.glob("0/*/user_*/poses_full.npy"))[0]
EXAMPLE_VIDEO = clip_directory(EXAMPLE_POSES) / "clip.mp4"
