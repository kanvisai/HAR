"""
Visualiza un archivo de poses .npy (shape: frames x 8 joints x xy).

Los datos están en ./videos/ (raíz del proyecto).

Uso (desde Tecnica_Heuristica_HAR/):
  python visualizar_npy.py
  python visualizar_npy.py 1/CLIP/user_136/poses_full.npy
  python visualizar_npy.py videos/1/CLIP/user_136/poses_full.npy
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_pose.config import DEFAULT_NPY_RELATIVE
from har_pose.io_utils import resolve_fps, resolve_npy_path

CONNECTIONS = [
    (0, 1),
    (0, 2), (2, 4),
    (1, 3), (3, 5),
    (0, 6), (1, 7),
    (6, 7),
]

W, H = 720, 720


def draw_skeleton(draw: ImageDraw.ImageDraw, points: np.ndarray) -> None:
    for i, j in CONNECTIONS:
        if i >= len(points) or j >= len(points):
            continue
        p1, p2 = points[i], points[j]
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        x1, y1 = int(p1[0] * W), int(p1[1] * H)
        x2, y2 = int(p2[0] * W), int(p2[1] * H)
        draw.line([(x1, y1), (x2, y2)], fill=(255, 220, 0), width=3)

    r = 5
    for k, pt in enumerate(points):
        if not np.isfinite(pt).all():
            continue
        x, y = int(pt[0] * W), int(pt[1] * H)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 140, 255))
        draw.text((x + 6, y - 6), str(k), fill=(200, 200, 200))


def visualize(npy_path: Path, fps: float = 20.0) -> None:
    data = np.load(npy_path)
    print(f"Archivo: {npy_path}")
    print(f"Shape:   {data.shape}  dtype={data.dtype}")

    if data.ndim == 3:
        n_frames, multi = len(data), False
    elif data.ndim == 4 and data.shape[1] == 2:
        n_frames, multi = len(data), True
    else:
        raise ValueError(f"Formato no soportado: {data.shape}. Esperado (T,J,2) o (T,2,J,2)")

    mask_path = npy_path.with_name("valid_mask.npy")
    valid_mask = np.load(mask_path) if mask_path.exists() else None
    if valid_mask is not None:
        print(f"valid_mask: {int(valid_mask.sum())}/{len(valid_mask)} frames válidos")

    delay_ms = max(1, int(1000 / fps))
    frame_idx = [0]
    paused = [False]

    root = tk.Tk()
    root.title(f"Pose NPY — {npy_path.name}")
    root.resizable(False, False)
    label = tk.Label(root)
    label.pack()
    hint = tk.Label(
        root,
        text="Espacio: pausa | ←/→: frame | Q/Esc: salir",
        font=("TkDefaultFont", 9),
    )
    hint.pack(pady=(0, 6))

    def build_frame(idx: int) -> Image.Image:
        img = Image.new("RGB", (W, H), (18, 18, 22))
        draw = ImageDraw.Draw(img)

        if multi:
            draw_skeleton(draw, data[idx, 0])
            draw_skeleton(draw, data[idx, 1])
            draw.text((10, 8), "2 usuarios", fill=(255, 255, 255))
        else:
            draw_skeleton(draw, data[idx])

        status = "OK" if (valid_mask is None or valid_mask[idx]) else "NaN/invalid"
        draw.text((10, 32), f"Frame {idx}/{n_frames - 1}  [{status}]", fill=(255, 255, 255))
        if paused[0]:
            draw.text((10, 56), "PAUSA", fill=(255, 80, 80))
        return img

    def refresh() -> None:
        idx = frame_idx[0] % n_frames
        photo = ImageTk.PhotoImage(build_frame(idx))
        label.config(image=photo)
        label.image = photo

    def tick() -> None:
        if not paused[0]:
            frame_idx[0] = (frame_idx[0] + 1) % n_frames
            refresh()
        root.after(delay_ms, tick)

    def on_key(event) -> None:
        key = event.keysym
        if key in ("q", "Q", "Escape"):
            root.destroy()
        elif key == "space":
            paused[0] = not paused[0]
            refresh()
        elif key == "Left":
            paused[0] = True
            frame_idx[0] = (frame_idx[0] - 1) % n_frames
            refresh()
        elif key == "Right":
            paused[0] = True
            frame_idx[0] = (frame_idx[0] + 1) % n_frames
            refresh()

    root.bind("<KeyPress>", on_key)
    refresh()
    root.after(delay_ms, tick)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza poses desde un .npy en videos/")
    parser.add_argument(
        "npy_path",
        nargs="?",
        default=None,
        help=f"Ruta relativa a videos/ (default: {DEFAULT_NPY_RELATIVE})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS (si se omite, meta.json o 20)",
    )
    args = parser.parse_args()
    path = resolve_npy_path(args.npy_path)
    try:
        fps, src = resolve_fps(path, args.fps)
        print(f"FPS: {fps} ({src})")
    except Exception:
        fps = args.fps if args.fps and args.fps > 0 else 20.0
        print(f"FPS: {fps} (fallback)")
    visualize(path, fps=fps)
