"""
Open a window with the skeleton and save the same video under temp_output.
That file is replaced on every run.

On the right, each action appears when it ends, with its time range.

Usage:
  python visualize_ske.py /path/3_npy/<clip>/poses_full.npy

Space: pause. Arrows: one frame. Q: quit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_actions.detect import actions_at_frame, detect
from har_actions.draw import join_list, play, recolor_active_skeleton, save_temp, skeleton_canvas
from har_actions.pose import EXAMPLE_POSES, fps_and_video, frame_aspect, load_poses, smooth

WIDTH, HEIGHT = 960, 720


def main() -> None:
    parser = argparse.ArgumentParser(description="Window with the skeleton and the action list.")
    parser.add_argument("poses", type=Path, nargs="?", default=EXAMPLE_POSES, help="poses_full.npy")
    args = parser.parse_args()

    if not args.poses.is_file():
        raise SystemExit(f"npy not found: {args.poses}")

    poses = smooth(load_poses(args.poses), window=3)
    fps, video = fps_and_video(args.poses)
    actions = detect(poses, fps, frame_aspect(video))
    temporary = ROOT / "temp_output" / "skeleton.mp4"
    print(f"Temporary: {temporary}")
    print("Space: pause. Arrows: one frame. Q: quit.")

    def build(index: int):
        current = actions_at_frame(actions, index)
        canvas, shifted = skeleton_canvas(poses[index], WIDTH, HEIGHT)
        recolor_active_skeleton(canvas, shifted, current)
        return join_list(canvas, actions, index, fps)

    save_temp(temporary, fps, len(poses), build)
    play("Skeleton", len(poses), fps, build)


if __name__ == "__main__":
    main()
