"""
Open the video window with the skeleton and also write into temp_output
(replacing the previous files on every run):

  actions.json
  skeleton.mp4
  overlay.mp4

On the right, each action appears when it ends, with its time range.

Usage:
  python visualize_sup.py /path/3_npy/<clip>/poses_full.npy

Space: pause. Arrows: one frame. Q: quit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect_json import write_json
from har_actions.detect import actions_at_frame, detect, report
from har_actions.draw import draw_skeleton, join_list, play, recolor_active_skeleton, save_temp, skeleton_canvas
from har_actions.pose import EXAMPLE_POSES, EXAMPLE_VIDEO, fps_and_video, frame_aspect, load_poses, smooth

WIDTH, HEIGHT = 960, 720


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video window and, in temp_output, the JSON, the skeleton and the overlay."
    )
    parser.add_argument("poses", type=Path, nargs="?", default=EXAMPLE_POSES, help="poses_full.npy")
    parser.add_argument("--video", type=Path, default=None, help="clip.mp4. If omitted, it is looked up from the npy path.")
    args = parser.parse_args()

    if not args.poses.is_file():
        raise SystemExit(f"npy not found: {args.poses}")

    poses = smooth(load_poses(args.poses), window=3)
    fps, video = fps_and_video(args.poses, args.video)
    if video is None:
        video = EXAMPLE_VIDEO
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")

    actions = detect(poses, fps, frame_aspect(video))
    output = ROOT / "temp_output"
    json_path = output / "actions.json"
    skeleton_path = output / "skeleton.mp4"
    overlay_path = output / "overlay.mp4"

    write_json(json_path, args.poses, fps, video, len(poses), actions)
    print(report(actions, fps))
    print()

    def build_skeleton(index: int):
        current = actions_at_frame(actions, index)
        canvas, shifted = skeleton_canvas(poses[index], WIDTH, HEIGHT)
        recolor_active_skeleton(canvas, shifted, current)
        return join_list(canvas, actions, index, fps)

    print(f"JSON:       {json_path}")
    print(f"Skeleton:   {skeleton_path}")
    save_temp(skeleton_path, fps, len(poses), build_skeleton)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {video}")

    n = len(poses)
    read_index = -1
    current_frame = None

    def build_overlay(index: int):
        nonlocal read_index, current_frame
        if index != read_index + 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            read_index = index - 1
        if index != read_index:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit(f"Could not read frame {index} of {video}")
            read_index = index
            current_frame = frame
        view = current_frame.copy()
        current = actions_at_frame(actions, index)
        draw_skeleton(view, poses[index], current, radius=7, thickness=3)
        return join_list(view, actions, index, fps)

    try:
        print(f"Overlay:    {overlay_path}")
        save_temp(overlay_path, fps, n, build_overlay)
        print("Space: pause. Arrows: one frame. Q: quit.")
        play("Video + skeleton", n, fps, build_overlay)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
