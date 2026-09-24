#!/usr/bin/env python3
"""
Validación in-situ: reproduce el esqueleto .npy y muestra las acciones
activas en cada frame (tiempo real / a velocidad FPS).

Uso (desde Tecnica_Heuristica_HAR/; datos en ./videos/):
  python validate_realtime.py
  python validate_realtime.py 1/CLIP/user_136/poses_full.npy
  python validate_realtime.py --save-txt
  python validate_realtime.py --filter WRIST_NEAR

Controles:
  Espacio     pausa / reanuda
  ← / →       frame anterior / siguiente (pausa)
  ↑ / ↓       ±10 frames
  Home/End    inicio / fin
  [ / ]       más lento / más rápido
  F           filtrar: solo CODE=0 / todos
  Q / Esc     salir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_pose.config import DEFAULT_NPY_RELATIVE, Segment
from har_pose.config import DetectionConfig
from har_pose.detectors import ActionAnalysisPipeline
from har_pose.io_utils import (
    active_segments_at_frame,
    default_output_path,
    load_pose_npy,
    resolve_fps,
    resolve_npy_path,
    seconds_to_timestamp,
    write_results_txt,
)

CONNECTIONS = [
    (0, 1),
    (0, 2), (2, 4),
    (1, 3), (3, 5),
    (0, 6), (1, 7),
    (6, 7),
]

# Panel: canvas esqueleto + lista de acciones
SKEL_W, SKEL_H = 640, 640
PANEL_W = 420
TOTAL_W = SKEL_W + PANEL_W
TOTAL_H = SKEL_H

CODE_COLOR = {
    0: (80, 220, 120),   # verde
    1: (240, 200, 60),   # amarillo
    2: (220, 80, 80),    # rojo
}


def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def draw_skeleton(draw: ImageDraw.ImageDraw, points: np.ndarray) -> None:
    for i, j in CONNECTIONS:
        if i >= len(points) or j >= len(points):
            continue
        p1, p2 = points[i], points[j]
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        x1, y1 = int(p1[0] * SKEL_W), int(p1[1] * SKEL_H)
        x2, y2 = int(p2[0] * SKEL_W), int(p2[1] * SKEL_H)
        draw.line([(x1, y1), (x2, y2)], fill=(255, 220, 0), width=3)
    r = 5
    for k, pt in enumerate(points):
        if not np.isfinite(pt).all():
            continue
        x, y = int(pt[0] * SKEL_W), int(pt[1] * SKEL_H)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 160, 255))


def draw_frame(
    poses: np.ndarray,
    frame: int,
    active: list[Segment],
    *,
    fps: float,
    paused: bool,
    speed: float,
    only_code0: bool,
    name_filter: str,
    n_frames: int,
) -> Image.Image:
    img = Image.new("RGB", (TOTAL_W, TOTAL_H), (18, 18, 22))
    draw = ImageDraw.Draw(img)

    # Fondo esqueleto
    draw.rectangle([0, 0, SKEL_W - 1, SKEL_H - 1], fill=(12, 12, 16))
    draw_skeleton(draw, poses[frame])

    font = _font(15)
    font_sm = _font(12)
    font_lg = _font(18)

    t = frame / fps
    header = (
        f"Frame {frame}/{n_frames - 1}   "
        f"{seconds_to_timestamp(t)}   "
        f"x{speed:.2f}   "
        f"{'PAUSA' if paused else 'PLAY'}"
    )
    draw.text((12, 10), header, fill=(255, 255, 255), font=font_lg)

    # Panel derecho
    x0 = SKEL_W + 12
    draw.rectangle([SKEL_W, 0, TOTAL_W, TOTAL_H], fill=(24, 24, 30))
    draw.line([(SKEL_W, 0), (SKEL_W, TOTAL_H)], fill=(60, 60, 70), width=2)
    draw.text((x0, 12), "ACCIONES ACTIVAS", fill=(230, 230, 230), font=font_lg)
    draw.text(
        (x0, 40),
        "CODE 0=verde  1=amarillo",
        fill=(160, 160, 170),
        font=font_sm,
    )

    # Filtrar visualización
    shown = active
    if only_code0:
        shown = [s for s in shown if s.code == 0]
    if name_filter:
        shown = [s for s in shown if name_filter.upper() in s.name.upper()]

    # Priorizar: acciones dinámicas (sin ARM_/WRIST_NEAR_ estáticos largos) arriba
    # Orden: code asc, luego nombre
    shown = sorted(shown, key=lambda s: (s.code, s.name))

    y = 70
    if not shown:
        draw.text((x0, y), "(ninguna con el filtro actual)", fill=(140, 140, 150), font=font)
    else:
        max_lines = 28
        for s in shown[:max_lines]:
            color = CODE_COLOR.get(s.code, (200, 200, 200))
            # chip
            draw.rectangle([x0, y, x0 + 10, y + 14], fill=color)
            label = s.name
            if len(label) > 34:
                label = label[:31] + "..."
            draw.text((x0 + 16, y - 1), f"{label}", fill=color, font=font)
            y += 20
            rem = s.end_frame - frame
            draw.text(
                (x0 + 16, y - 2),
                f"code={s.code}  queda ~{rem / fps:.2f}s",
                fill=(130, 130, 140),
                font=font_sm,
            )
            y += 18
            if y > TOTAL_H - 80:
                draw.text((x0, y), f"... +{len(shown) - max_lines} más", fill=(140, 140, 150), font=font_sm)
                break

    # Leyenda controles
    draw.text(
        (x0, TOTAL_H - 55),
        "Espacio pausa  ←→ frame  [] velocidad",
        fill=(120, 120, 130),
        font=font_sm,
    )
    draw.text(
        (x0, TOTAL_H - 35),
        "F solo CODE0   Q salir",
        fill=(120, 120, 130),
        font=font_sm,
    )
    return img


def run_viewer(
    npy_path: Path,
    fps: float,
    *,
    save_txt: bool = False,
    name_filter: str = "",
    output_dir: Path | None = None,
) -> None:
    poses = load_pose_npy(npy_path)
    if poses.ndim == 4:
        poses = poses[:, 0]

    print(f"Cargando detección… ({poses.shape[0]} frames @ {fps} fps)")
    cfg = DetectionConfig()
    segments, _feats = ActionAnalysisPipeline(cfg).run(poses, fps)
    print(f"Segmentos detectados: {len(segments)}")

    if save_txt:
        out_path = default_output_path(npy_path, output_dir)
        out_path = out_path.with_name(out_path.name.replace("_actions.txt", "_actions.txt"))
        write_results_txt(
            out_path,
            segments,
            source=str(npy_path.resolve()),
            fps=fps,
            config=cfg,
            extra_header=[f"num_segments={len(segments)}", "from=validate_realtime"],
        )
        print(f"TXT guardado: {out_path}")

    n_frames = len(poses)
    delay0 = max(1, int(1000 / fps))
    state = {
        "frame": 0,
        "paused": False,
        "speed": 1.0,
        "only_code0": False,
        "filter": name_filter,
    }

    root = tk.Tk()
    root.title(f"Validación — {npy_path.name}")
    root.resizable(False, False)
    label = tk.Label(root)
    label.pack()

    photo_ref = {"img": None}

    def refresh() -> None:
        f = state["frame"] % n_frames
        active = active_segments_at_frame(segments, f)
        img = draw_frame(
            poses,
            f,
            active,
            fps=fps,
            paused=state["paused"],
            speed=state["speed"],
            only_code0=state["only_code0"],
            name_filter=state["filter"],
            n_frames=n_frames,
        )
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        photo_ref["img"] = photo

    def tick() -> None:
        if not state["paused"]:
            state["frame"] = (state["frame"] + 1) % n_frames
            refresh()
        delay = max(1, int(delay0 / max(0.1, state["speed"])))
        root.after(delay, tick)

    def on_key(event) -> None:
        key = event.keysym
        if key in ("q", "Q", "Escape"):
            root.destroy()
            return
        if key == "space":
            state["paused"] = not state["paused"]
        elif key == "Left":
            state["paused"] = True
            state["frame"] = (state["frame"] - 1) % n_frames
        elif key == "Right":
            state["paused"] = True
            state["frame"] = (state["frame"] + 1) % n_frames
        elif key == "Up":
            state["paused"] = True
            state["frame"] = (state["frame"] + 10) % n_frames
        elif key == "Down":
            state["paused"] = True
            state["frame"] = (state["frame"] - 10) % n_frames
        elif key == "Home":
            state["paused"] = True
            state["frame"] = 0
        elif key == "End":
            state["paused"] = True
            state["frame"] = n_frames - 1
        elif key == "bracketleft":
            state["speed"] = max(0.25, state["speed"] / 1.25)
        elif key == "bracketright":
            state["speed"] = min(4.0, state["speed"] * 1.25)
        elif key in ("f", "F"):
            state["only_code0"] = not state["only_code0"]
        refresh()

    root.bind("<KeyPress>", on_key)
    refresh()
    root.after(delay0, tick)
    root.mainloop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualiza .npy + acciones detectadas in-situ")
    p.add_argument(
        "npy",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Ruta al .npy relativa a videos/ "
            f"(default: {DEFAULT_NPY_RELATIVE})"
        ),
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS (si se omite, se lee de meta.json junto al clip)",
    )
    p.add_argument(
        "--save-txt",
        action="store_true",
        help="También escribe el .txt en output/<CLIP_NAME>/",
    )
    p.add_argument(
        "--filter",
        type=str,
        default="",
        help="Mostrar solo acciones cuyo nombre contenga este texto (ej: WRIST_NEAR)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Raíz de salida si usas --save-txt",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        npy = resolve_npy_path(args.npy)
    except Exception as e:
        print(f"Error resolviendo .npy: {e}", file=sys.stderr)
        return 2
    print(f"NPY: {npy}")
    try:
        fps, fps_src = resolve_fps(npy, args.fps)
    except Exception as e:
        print(f"Error resolviendo FPS: {e}", file=sys.stderr)
        return 2
    print(f"FPS: {fps}  (origen: {fps_src})")
    out_dir = Path(args.output_dir) if args.output_dir else None
    run_viewer(
        npy,
        fps,
        save_txt=args.save_txt,
        name_filter=args.filter,
        output_dir=out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
