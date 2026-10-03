"""
Extract an 8-point upper-body skeleton from a clip with YOLO pose.

Points, in order:
  0 left shoulder, 1 right shoulder,
  2 left elbow, 3 right elbow,
  4 left wrist, 5 right wrist,
  6 left hip, 7 right hip.

Left and right are the person's, not the image's.
A point YOLO does not see (confidence under the threshold) is stored as NaN
and is not drawn. Nothing is filled in from the other arm or from nearby frames.

Writes, next to the clip by default:
  skeleton.npy   float32, shape (frames, 8, 2), coordinates 0–1, y down
  overlay.mp4    the clip with the visible skeleton on top

Usage:
  python extract_skeleton.py /path/to/clip.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# COCO pose indexes used by YOLO.
COCO = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}
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
BONES = (
    (0, 1),
    (0, 2),
    (2, 4),
    (1, 3),
    (3, 5),
    (0, 6),
    (1, 7),
    (6, 7),
)
# BGR
JOINT_COLOR = (
    (255, 80, 0),
    (0, 80, 255),
    (0, 200, 0),
    (0, 255, 180),
    (255, 0, 0),
    (0, 140, 255),
    (0, 220, 220),
    (220, 220, 0),
)


def largest_person(result):
    """The closest person in the frame, measured by the box area. None if nobody is seen."""
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    areas = []
    for box in result.boxes.xyxy.cpu().numpy():
        areas.append(float((box[2] - box[0]) * (box[3] - box[1])))
    return int(np.argmax(areas))


def eight_points(result, index: int, width: int, height: int, min_conf: float) -> np.ndarray:
    """(8, 2) in 0–1. Unseen points are NaN."""
    points = np.full((8, 2), np.nan, dtype=np.float32)
    if index is None:
        return points
    raw = result.keypoints.data[index].cpu().numpy()
    for slot, name in enumerate(NAMES):
        x, y, conf = raw[COCO[name]]
        if conf < min_conf:
            continue
        if not (0 <= x < width and 0 <= y < height):
            continue
        points[slot, 0] = x / width
        points[slot, 1] = y / height
    return points


def draw_skeleton(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Draw only the joints that were seen. No line is drawn to a missing joint."""
    out = frame.copy()
    height, width = out.shape[:2]
    pixels = []
    for x, y in points:
        if not np.isfinite(x) or not np.isfinite(y):
            pixels.append(None)
            continue
        pixels.append((int(round(x * width)), int(round(y * height))))
    for a, b in BONES:
        if pixels[a] is None or pixels[b] is None:
            continue
        cv2.line(out, pixels[a], pixels[b], (255, 255, 255), 2, cv2.LINE_AA)
    for joint, pixel in enumerate(pixels):
        if pixel is None:
            continue
        cv2.circle(out, pixel, 5, JOINT_COLOR[joint], -1, cv2.LINE_AA)
    return out


def read_poses(model: YOLO, video_path: Path, min_conf: float) -> np.ndarray:
    """(frames, 8, 2). Unseen points are NaN. No overlay is written."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se puede abrir el vídeo: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    poses = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(frame, verbose=False, conf=0.25)[0]
        person = largest_person(result)
        poses.append(eight_points(result, person, width, height, min_conf))
    capture.release()
    if not poses:
        return np.zeros((0, 8, 2), dtype=np.float32)
    return np.stack(poses)


def extract_sets(root: Path, min_conf: float, model_name: str) -> None:
    """0/, 3/ and 4/ become 0_npy, 3_npy and 4_npy, each as <clip>/poses_full.npy."""
    model = YOLO(model_name)
    for source_name, dest_name in (("0", "0_npy"), ("3", "3_npy"), ("4", "4_npy")):
        folders = sorted(
            folder
            for folder in (root / source_name).iterdir()
            if folder.is_dir() and (folder / "clip.mp4").is_file()
        )
        for number, folder in enumerate(folders, start=1):
            destination = root / dest_name / folder.name / "poses_full.npy"
            if destination.is_file():
                print(f"[{source_name} {number}/{len(folders)}] ya está {folder.name}")
                continue
            print(f"[{source_name} {number}/{len(folders)}] {folder.name}", flush=True)
            poses = read_poses(model, folder / "clip.mp4", min_conf)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, poses)
            print(f"  {tuple(poses.shape)} -> {destination}", flush=True)


def extract(video_path: Path, output_dir: Path, min_conf: float, model_name: str) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"No se puede abrir el vídeo: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 12.5
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "overlay.mp4"
    skeleton_path = output_dir / "skeleton.npy"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    model = YOLO(model_name)
    poses = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(frame, verbose=False, conf=0.25)[0]
        person = largest_person(result)
        points = eight_points(result, person, width, height, min_conf)
        poses.append(points)
        writer.write(draw_skeleton(frame, points))
        index += 1
        if index % 25 == 0 or index == frame_count:
            print(f"  {index}/{frame_count or '?'}")

    capture.release()
    writer.release()
    skeleton = np.stack(poses) if poses else np.zeros((0, 8, 2), dtype=np.float32)
    np.save(skeleton_path, skeleton)

    seen = np.isfinite(skeleton).all(axis=2)
    print(f"Vídeo:     {video_path}")
    print(f"Frames:    {len(skeleton)}   fps: {fps:.2f}   {width}x{height}")
    print(f"Puntos:    {', '.join(NAMES)}")
    print(f"Visibles:  {int(seen.sum())} de {seen.size}  (umbral de confianza {min_conf})")
    print(f"Esqueleto: {skeleton_path}")
    print(f"Overlay:   {overlay_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Esqueleto de 8 puntos con YOLO, sin rellenar lo que no se ve.")
    parser.add_argument("video", type=Path, nargs="?", help="clip.mp4. Sin esto, y con --sets, se procesan las carpetas 3 y 4.")
    parser.add_argument("--sets", action="store_true", help="Escribe 3_npy y 4_npy, solo poses_full.npy.")
    parser.add_argument("--out", type=Path, default=None, help="Carpeta de salida. Por defecto, example/ junto a este script.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confianza mínima de un punto. Por debajo, no se guarda ni se dibuja.")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "yolo11x-pose.pt"),
        help="Modelo YOLO pose.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.sets:
        extract_sets(root, args.conf, args.model)
        return
    if args.video is None or not args.video.is_file():
        raise SystemExit(f"No está el vídeo: {args.video}")
    output = args.out if args.out is not None else root / "example"
    extract(args.video, output, args.conf, args.model)


if __name__ == "__main__":
    main()
