#!/usr/bin/env python3
"""
CLI: detectar estados/acciones en una secuencia de pose .npy

Los datos viven en ./videos/ (relativo a la raíz del proyecto).

Ejemplos (desde Tecnica_Heuristica_HAR/):
  python detect_actions.py
  python detect_actions.py 1/12Diciembre2025_..._1/user_136/poses_full.npy
  python detect_actions.py videos/1/.../user_136/poses_full.npy
  python detect_actions.py --fps 25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permitir ejecución directa desde la raíz del proyecto
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_pose.config import DEFAULT_NPY_RELATIVE, DEFAULT_OUTPUT_DIR
from har_pose.config import DetectionConfig
from har_pose.detectors import ActionAnalysisPipeline
from har_pose.io_utils import (
    default_output_path,
    load_pose_npy,
    resolve_fps,
    resolve_npy_path,
    seconds_to_timestamp,
    write_debug_features,
    write_results_txt,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detecta estados/acciones corporales en un .npy de poses 2D"
    )
    p.add_argument(
        "npy",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Ruta al .npy. Relativa a videos/ por defecto "
            f"(ej: {DEFAULT_NPY_RELATIVE}). Si se omite, usa ese ejemplo."
        ),
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS (si se omite, se lee de meta.json junto al clip)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Ruta exacta del .txt (si no se indica: output/<CLIP_NAME>/..._actions.txt)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Raíz de salida; se crea subcarpeta con el nombre del clip ({DEFAULT_OUTPUT_DIR}/<CLIP>/)",
    )
    p.add_argument("--debug", action="store_true", help="Guarda features CSV y config JSON")
    p.add_argument(
        "--plot",
        action="store_true",
        help="Genera gráficos matplotlib de features + segmentos (opcional)",
    )
    p.add_argument(
        "--min-state-s",
        type=float,
        default=None,
        help="Sobrescribe min_state_duration_s",
    )
    p.add_argument(
        "--min-action-s",
        type=float,
        default=None,
        help="Sobrescribe min_action_duration_s",
    )
    return p.parse_args(argv)


def maybe_plot(feats, segments, out_png: Path, fps: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] --plot no disponible (matplotlib): {e}", file=sys.stderr)
        return

    keys = [
        ("elbow_angle_L", "Elbow L (deg)"),
        ("elbow_angle_R", "Elbow R (deg)"),
        ("wrist_waist_L", "Wrist-waist L"),
        ("wrist_chest_L", "Wrist-chest L"),
        ("shoulder_width", "Shoulder width (norm)"),
        ("frontal_ratio", "Frontal ratio"),
    ]
    t = np_time = __import__("numpy").arange(feats.num_frames) / fps
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2.2 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, (k, title) in zip(axes, keys):
        ax.plot(t, feats.get(k), lw=1.0)
        ax.set_ylabel(title, fontsize=8)
        ax.grid(True, alpha=0.3)
        for seg in segments:
            if seg.feature and k in seg.feature.replace("&", " ").split():
                ax.axvspan(seg.start_s(fps), seg.end_s(fps), alpha=0.15, color="C1")
            elif seg.name.startswith("ARM_EXTENDED") and "elbow_angle" in k:
                side = "L" if "LEFT" in seg.name else "R"
                if k.endswith(side):
                    ax.axvspan(seg.start_s(fps), seg.end_s(fps), alpha=0.12, color="C2")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("HAR pose features")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Plot guardado: {out_png}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        npy_path = resolve_npy_path(args.npy)
    except Exception as e:
        print(f"Error resolviendo .npy: {e}", file=sys.stderr)
        return 2
    print(f"NPY: {npy_path}")

    try:
        fps, fps_src = resolve_fps(npy_path, args.fps)
    except Exception as e:
        print(f"Error resolviendo FPS: {e}", file=sys.stderr)
        return 2
    print(f"FPS: {fps}  (origen: {fps_src})")

    try:
        poses = load_pose_npy(npy_path)
    except Exception as e:
        print(f"Error cargando .npy: {e}", file=sys.stderr)
        return 2

    cfg = DetectionConfig()
    if args.min_state_s is not None:
        cfg.min_state_duration_s = args.min_state_s
    if args.min_action_s is not None:
        cfg.min_action_duration_s = args.min_action_s

    pipeline = ActionAnalysisPipeline(cfg)
    segments, feats = pipeline.run(poses, fps)

    out_dir = Path(args.output_dir)
    out_path = Path(args.output) if args.output else default_output_path(npy_path, out_dir)
    if args.output is None:
        out_path = out_path.with_name(out_path.name.replace("_actions.txt", "_actions.txt"))

    write_results_txt(
        out_path,
        segments,
        source=str(npy_path.resolve()),
        fps=fps,
        config=cfg,
        extra_header=[
            f"num_frames={poses.shape[0]}",
            f"duration_s={poses.shape[0] / fps:.3f}",
            f"num_segments={len(segments)}",
            "config_variant=stable",
            f"fps_source={fps_src}",
        ],
    )
    print(f"Escrito: {out_path}")
    print(f"Segmentos: {len(segments)}")
    if segments:
        preview = segments[:8]
        for s in preview:
            print(
                f"  {s.name:28s} "
                f"{seconds_to_timestamp(s.start_s(fps))} → "
                f"{seconds_to_timestamp(s.end_s(fps))}  code={s.code}"
            )
        if len(segments) > 8:
            print(f"  ... (+{len(segments) - 8} más)")

    if args.debug:
        dbg_csv = out_path.with_suffix(".features.csv")
        dbg_json = out_path.with_suffix(".config.json")
        write_debug_features(
            dbg_csv,
            feats.values,
            fps,
            keys=[
                "elbow_angle_L",
                "elbow_angle_R",
                "wrist_waist_L",
                "wrist_waist_R",
                "wrist_chest_L",
                "wrist_chest_R",
                "wrists_apart",
                "torso_lean",
                "frontal_ratio",
                "shoulder_width",
                "speed_wrist_L",
                "speed_wrist_R",
            ],
        )
        dbg_json.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        reasons = out_path.with_suffix(".reasons.txt")
        with reasons.open("w", encoding="utf-8") as f:
            for s in segments:
                f.write(
                    f"{s.name} | frames {s.start_frame}-{s.end_frame} | "
                    f"feature={s.feature} | enter={s.enter_threshold} | "
                    f"exit={s.exit_threshold} | {s.reason}\n"
                )
        print(f"Debug: {dbg_csv}, {dbg_json}, {reasons}")

    if args.plot:
        maybe_plot(feats, segments, out_path.with_suffix(".png"), fps)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
