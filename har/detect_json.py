"""
Read a poses_full.npy and write the gesture JSON.
Also leaves a copy at temp_output/actions.json, replaced on every run.

Usage:
  python detect_json.py /path/3_npy/<clip>/poses_full.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_actions.detect import NOT_OBSERVABLE, detect, disabled_actions, report
from har_actions.pose import EXAMPLE_POSES, NAMES, fps_and_video, frame_aspect, load_poses, smooth


def write_json(destination: Path, poses_path: Path, fps: float, video: Path | None, n_frames: int, actions) -> dict:
    document = {
        "poses": str(poses_path),
        "video": str(video) if video else None,
        "fps": fps,
        "n_frames": int(n_frames),
        "duration_s": round(n_frames / fps, 2),
        "keypoints": list(NAMES),
        "note": (
            "Actions: extend the arm, extend the forearm, retract the arm, turn the torso, "
            "crouch a little, and a steady wrist-to-waist distance. "
            "A point missing from the npy is not measured."
        ),
        "not_observable_with_this_pose": list(NOT_OBSERVABLE),
        "disabled_actions": disabled_actions(),
        "actions": [action.to_dict(fps) for action in actions],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect gestures and write a JSON file.")
    parser.add_argument("poses", type=Path, nargs="?", default=EXAMPLE_POSES, help="poses_full.npy")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON. Default: actions.json next to the npy.")
    args = parser.parse_args()

    if not args.poses.is_file():
        raise SystemExit(f"npy not found: {args.poses}")

    output = args.output if args.output is not None else args.poses.with_name("actions.json")
    poses = smooth(load_poses(args.poses), window=3)
    fps, video = fps_and_video(args.poses)
    actions = detect(poses, fps, frame_aspect(video))

    document = write_json(output, args.poses, fps, video, len(poses), actions)
    temporary = ROOT / "temp_output" / "actions.json"
    if temporary.resolve() != output.resolve():
        write_json(temporary, args.poses, fps, video, len(poses), actions)

    print(f"Poses:    {args.poses}")
    print(f"Video:    {video}")
    print(f"Frames:   {len(poses)}   fps: {fps}   duration: {document['duration_s']} s")
    disabled = document["disabled_actions"]
    if disabled:
        print("Disabled in actions_config.json: " + ", ".join(disabled))
    print()
    print(report(actions, fps))
    print()
    print(f"JSON:       {output}")
    print(f"Temporary:  {temporary}")


if __name__ == "__main__":
    main()
