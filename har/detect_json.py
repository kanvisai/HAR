"""
Read a poses_full.npy and write the gesture JSON.
Also leaves a copy at temp_output/actions.json, replaced on every run.
The category evaluation is written to eval_cat_output.json.
summary_output.json is 1 when the category is inconclusive and 0 for c0, c1, c5 or c100.

Usage:
  python detect_json.py /path/videos/<cat>/<clip>/user_N/poses_full.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_actions.detect import NOT_OBSERVABLE, detect, disabled_actions, evaluate_category, report
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
            "Actions: walk, stop, turn the torso, swing the arm, carry, hand up, "
            "and hand to the body. "
            "Categories 0, 1 and 5. A point missing from the npy is not measured. "
            "Looking down is not measured: the skeleton has no head point."
        ),
        "not_observable_with_this_pose": list(NOT_OBSERVABLE),
        "disabled_actions": disabled_actions(),
        "actions": [action.to_dict(fps) for action in actions],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


def write_eval(destination: Path, poses_path: Path, fps: float, video: Path | None, poses, actions) -> dict:
    document = {
        "poses": str(poses_path),
        "video": str(video) if video else None,
        "fps": fps,
        "n_frames": int(len(poses)),
        "duration_s": round(len(poses) / fps, 2),
    }
    document.update(evaluate_category(poses, actions, fps))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


DETECTED_CATEGORIES = {"c0", "c1", "c5", "c100"}


def write_summary(destination: Path, poses_path: Path, evaluation: dict) -> dict:
    """1 when the category is inconclusive, 0 for c0, c1, c5 or c100."""
    category = evaluation["category"]
    document = {
        "poses": str(poses_path),
        "category": category,
        "output": 0 if category in DETECTED_CATEGORIES else 1,
        "note": "output is 0 for c0, c1, c5 or c100, and 1 for inconclusive.",
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
    evaluation = write_eval(output.with_name("eval_cat_output.json"), args.poses, fps, video, poses, actions)
    summary = write_summary(output.with_name("summary_output.json"), args.poses, evaluation)
    temporary = ROOT / "temp_output" / "actions.json"
    temporary_eval = ROOT / "temp_output" / "eval_cat_output.json"
    temporary_summary = ROOT / "temp_output" / "summary_output.json"
    if temporary.resolve() != output.resolve():
        write_json(temporary, args.poses, fps, video, len(poses), actions)
    if temporary_eval.resolve() != output.with_name("eval_cat_output.json").resolve():
        write_eval(temporary_eval, args.poses, fps, video, poses, actions)
    if temporary_summary.resolve() != output.with_name("summary_output.json").resolve():
        write_summary(temporary_summary, args.poses, evaluation)

    print(f"Poses:    {args.poses}")
    print(f"Video:    {video}")
    print(f"Frames:   {len(poses)}   fps: {fps}   duration: {document['duration_s']} s")
    disabled = document["disabled_actions"]
    if disabled:
        print("Disabled in actions_config.json: " + ", ".join(disabled))
    print()
    print(report(actions, fps))
    print()
    print(f"Category:   {evaluation['category']}")
    print(f"            {evaluation['reason']}")
    print(f"Summary:    {summary['output']}")
    print()
    print(f"JSON:       {output}")
    print(f"Evaluation: {output.with_name('eval_cat_output.json')}")
    print(f"Summary file: {output.with_name('summary_output.json')}")
    print(f"Temporary:  {temporary}")


if __name__ == "__main__":
    main()
