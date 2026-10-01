"""
Read a poses_full.npy and write the gesture JSON.
Also leaves a copy at temp_output/actions.json, replaced on every run.

Usage:
  python detect_json.py /path/user_x/poses_full.npy
  python detect_json.py /path/user_x/poses_full.npy -o /path/actions.json
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
from har_actions.pose import EXAMPLE_POSES, NAMES, fps_and_video, load_poses, smooth


def write_json(destination: Path, poses_path: Path, fps: float, video: Path | None, n_frames: int, actions) -> dict:
    """Write the action JSON to destination, replacing the file if it already exists."""
    document = {
        "poses": str(poses_path),
        "video": str(video) if video else None,
        "fps": fps,
        "n_frames": int(n_frames),
        "duration_s": round(n_frames / fps, 2),
        "keypoints": list(NAMES),
        "note": (
            "Each action is an arm or torso gesture after the product is already in hand. "
            "With 8 points the object is not visible: the JSON marks the movement consistent "
            "with concealing it, so it can be checked on the video."
        ),
        "not_observable_with_this_pose": list(NOT_OBSERVABLE),
        "disabled_actions": disabled_actions(),
        "actions": [action.to_dict(fps) for action in actions],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect concealment gestures and write a JSON file.")
    parser.add_argument("poses", type=Path, nargs="?", default=EXAMPLE_POSES, help="poses_full.npy")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON. Default: actions.json next to the npy.")
    args = parser.parse_args()

    if not args.poses.is_file():
        raise SystemExit(f"npy not found: {args.poses}")

    output = args.output if args.output is not None else args.poses.with_name("actions.json")
    raw = load_poses(args.poses)
    poses = smooth(raw, window=3)
    fps, video = fps_and_video(args.poses)
    actions = detect(poses, fps)

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
