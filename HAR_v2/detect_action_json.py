#!/usr/bin/env python3
"""
Detecta TODAS las acciones/estados y escribe un JSON.

Uso:
  python detect_action_json.py
  python detect_action_json.py 2/CLIP/user_603/poses_full.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_pose.config import (
    DEFAULT_NPY_RELATIVE,
    DEFAULT_OUTPUT_DIR,
    UNSUPPORTED_2D,
)
from har_pose.config import DetectionConfig
from har_pose.detectors import ActionAnalysisPipeline
from har_pose.io_utils import (
    default_output_path,
    load_pose_npy,
    resolve_fps,
    resolve_npy_path,
    seconds_to_timestamp,
)


def segments_to_jsonable(segments, fps: float) -> list[dict[str, Any]]:
    rows = []
    for s in sorted(segments, key=lambda x: (x.start_frame, x.name)):
        rows.append(
            {
                "action": s.name,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start": seconds_to_timestamp(s.start_s(fps)),
                "end": seconds_to_timestamp(s.end_s(fps)),
                "start_s": round(s.start_s(fps), 6),
                "end_s": round(s.end_s(fps), 6),
                "duration_s": round(s.duration_s(fps), 6),
                "code": s.code,
                "feature": s.feature,
                "reason": s.reason,
                "enter_threshold": s.enter_threshold,
                "exit_threshold": s.exit_threshold,
            }
        )
    return rows


def default_json_path(npy_path: Path, output_dir: Path | None = None) -> Path:
    txt_path = default_output_path(npy_path, output_dir)
    return txt_path.with_name(txt_path.name.replace("_actions.txt", "_actions.json"))


def build_payload(
    *,
    npy_path: Path,
    fps: float,
    fps_source: str,
    poses_shape: tuple,
    segments,
) -> dict[str, Any]:
    return {
        "source": str(npy_path.resolve()),
        "fps": fps,
        "fps_source": fps_source,
        "num_frames": int(poses_shape[0]),
        "duration_s": round(poses_shape[0] / fps, 6),
        "filter": None,
        "config_variant": "stable",
        "num_segments": len(segments),
        "segments": segments_to_jsonable(segments, fps),
        "unsupported_2d": [{"action": name, "code": 2} for name in UNSUPPORTED_2D],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detecta todas las acciones/estados → JSON"
    )
    p.add_argument(
        "npy",
        type=str,
        nargs="?",
        default=None,
        help=f"Ruta relativa a videos/ (default: {DEFAULT_NPY_RELATIVE})",
    )
    p.add_argument("--fps", type=float, default=None, help="FPS (default: meta.json)")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Ruta del JSON (default: output/<clase>/<CLIP>/..._actions.json)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Raíz de salida (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Imprime el JSON por stdout",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="No escribe fichero (útil con --stdout)",
    )
    p.add_argument("--indent", type=int, default=2, help="Indentación JSON (default 2)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        npy_path = resolve_npy_path(args.npy)
    except Exception as e:
        print(f"Error resolviendo .npy: {e}", file=sys.stderr)
        return 2
    print(f"NPY: {npy_path}", file=sys.stderr)

    try:
        fps, fps_src = resolve_fps(npy_path, args.fps)
    except Exception as e:
        print(f"Error resolviendo FPS: {e}", file=sys.stderr)
        return 2
    print(f"FPS: {fps}  (origen: {fps_src})", file=sys.stderr)

    try:
        poses = load_pose_npy(npy_path)
    except Exception as e:
        print(f"Error cargando .npy: {e}", file=sys.stderr)
        return 2

    cfg = DetectionConfig()
    segments, _ = ActionAnalysisPipeline(cfg).run(poses, fps)
    print(f"Segmentos: {len(segments)}", file=sys.stderr)

    payload = build_payload(
        npy_path=npy_path,
        fps=fps,
        fps_source=fps_src,
        poses_shape=poses.shape,
        segments=segments,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=args.indent) + "\n"

    if not args.no_save:
        out_path = (
            Path(args.output)
            if args.output
            else default_json_path(npy_path, Path(args.output_dir))
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Escrito: {out_path}", file=sys.stderr)

    if args.stdout or args.no_save:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
