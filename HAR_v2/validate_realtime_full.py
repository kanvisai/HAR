#!/usr/bin/env python3
"""
Validación in-situ FULL: superpone el esqueleto .npy sobre el clip.mp4
y muestra el panel de acciones activas (igual que validate_realtime).

Estructura esperada:
  videos/<clase>/<CLIP>/clip.mp4
  videos/<clase>/<CLIP>/user_XXX/poses_full.npy

Uso (desde Tecnica_Heuristica_HAR/):
  python validate_realtime_full.py
  python validate_realtime_full.py 1/CLIP/user_136/poses_full.npy
  python validate_realtime_full.py --mp4 1/CLIP/clip.mp4
  python validate_realtime_full.py --save-txt --filter WRIST_NEAR

Controles:
  Espacio     pausa / reanuda
  ← / →       frame anterior / siguiente
  ↑ / ↓       ±10 frames
  Home/End    inicio / fin
  [ / ]       más lento / más rápido
  F           solo CODE=0 / todos
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

try:
    import cv2
except ImportError:
    print("Se necesita OpenCV: pip install opencv-python", file=sys.stderr)
    raise SystemExit(1)

from har_pose.config import DEFAULT_NPY_RELATIVE, Segment
from har_pose.config import DetectionConfig
from har_pose.detectors import ActionAnalysisPipeline
from har_pose.io_utils import (
    active_segments_at_frame,
    default_output_path,
    load_pose_npy,
    resolve_clip_mp4,
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

# Vídeo a la izquierda + panel de acciones
VID_MAX_W = 960
VID_MAX_H = 540
PANEL_W = 420

CODE_COLOR = {
    0: (80, 220, 120),
    1: (240, 200, 60),
    2: (220, 80, 80),
}


def _font(size: int = 14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def draw_skeleton_on(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    width: int,
    height: int,
    *,
    ox: int = 0,
    oy: int = 0,
) -> None:
    """Dibuja pose normalizada [0,1] sobre un rectángulo width×height."""
    for i, j in CONNECTIONS:
        if i >= len(points) or j >= len(points):
            continue
        p1, p2 = points[i], points[j]
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        x1 = ox + int(p1[0] * width)
        y1 = oy + int(p1[1] * height)
        x2 = ox + int(p2[0] * width)
        y2 = oy + int(p2[1] * height)
        draw.line([(x1, y1), (x2, y2)], fill=(255, 220, 0), width=3)
    r = 5
    for pt in points:
        if not np.isfinite(pt).all():
            continue
        x = ox + int(pt[0] * width)
        y = oy + int(pt[1] * height)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 160, 255))


def fit_size(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    scale = min(max_w / src_w, max_h / src_h, 1.0)
    return max(1, int(src_w * scale)), max(1, int(src_h * scale))


class VideoReader:
    """Lectura de frames con seek perezoso."""

    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el vídeo: {path}")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._last_idx = -2
        self._last_bgr: np.ndarray | None = None

    def get_rgb(self, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, max(0, self.n_frames - 1)))
        if self._last_bgr is not None and idx == self._last_idx:
            bgr = self._last_bgr
        elif self._last_bgr is not None and idx == self._last_idx + 1:
            ok, bgr = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, bgr = self.cap.read()
                if not ok:
                    raise RuntimeError(f"No se pudo leer frame {idx}")
            self._last_idx = idx
            self._last_bgr = bgr
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = self.cap.read()
            if not ok:
                raise RuntimeError(f"No se pudo leer frame {idx}")
            self._last_idx = idx
            self._last_bgr = bgr
        return cv2.cvtColor(self._last_bgr, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        self.cap.release()


def draw_composite(
    video_rgb: np.ndarray,
    pose: np.ndarray | None,
    frame: int,
    active: list[Segment],
    *,
    fps: float,
    paused: bool,
    speed: float,
    only_code0: bool,
    name_filter: str,
    n_frames: int,
    disp_w: int,
    disp_h: int,
) -> Image.Image:
    total_w = disp_w + PANEL_W
    total_h = max(disp_h, 540)
    canvas = Image.new("RGB", (total_w, total_h), (18, 18, 22))

    # Vídeo redimensionado
    vid = Image.fromarray(video_rgb).resize((disp_w, disp_h), Image.BILINEAR)
    canvas.paste(vid, (0, 0))
    draw = ImageDraw.Draw(canvas)

    if pose is not None:
        draw_skeleton_on(draw, pose, disp_w, disp_h)

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
    # sombra del texto sobre el vídeo
    draw.text((11, 9), header, fill=(0, 0, 0), font=font_lg)
    draw.text((12, 10), header, fill=(255, 255, 255), font=font_lg)

    # Panel derecho
    x0 = disp_w + 12
    draw.rectangle([disp_w, 0, total_w, total_h], fill=(24, 24, 30))
    draw.line([(disp_w, 0), (disp_w, total_h)], fill=(60, 60, 70), width=2)
    draw.text((x0, 12), "ACCIONES ACTIVAS", fill=(230, 230, 230), font=font_lg)
    draw.text((x0, 40), "CODE 0=verde  1=amarillo", fill=(160, 160, 170), font=font_sm)

    shown = active
    if only_code0:
        shown = [s for s in shown if s.code == 0]
    if name_filter:
        shown = [s for s in shown if name_filter.upper() in s.name.upper()]
    shown = sorted(shown, key=lambda s: (s.code, s.name))

    y = 70
    if not shown:
        draw.text((x0, y), "(ninguna con el filtro actual)", fill=(140, 140, 150), font=font)
    else:
        max_lines = 28
        for i, s in enumerate(shown[:max_lines]):
            color = CODE_COLOR.get(s.code, (200, 200, 200))
            draw.rectangle([x0, y, x0 + 10, y + 14], fill=color)
            label = s.name if len(s.name) <= 34 else s.name[:31] + "..."
            draw.text((x0 + 16, y - 1), label, fill=color, font=font)
            y += 20
            rem = s.end_frame - frame
            draw.text(
                (x0 + 16, y - 2),
                f"code={s.code}  queda ~{rem / fps:.2f}s",
                fill=(130, 130, 140),
                font=font_sm,
            )
            y += 18
            if y > total_h - 80:
                rest = len(shown) - (i + 1)
                if rest > 0:
                    draw.text((x0, y), f"... +{rest} más", fill=(140, 140, 150), font=font_sm)
                break

    draw.text(
        (x0, total_h - 55),
        "Espacio pausa  ←→ frame  [] velocidad",
        fill=(120, 120, 130),
        font=font_sm,
    )
    draw.text(
        (x0, total_h - 35),
        "F solo CODE0   Q salir",
        fill=(120, 120, 130),
        font=font_sm,
    )
    return canvas


def run_viewer(
    npy_path: Path,
    mp4_path: Path,
    fps: float,
    *,
    save_txt: bool = False,
    name_filter: str = "",
    output_dir: Path | None = None,
) -> None:
    poses = load_pose_npy(npy_path)
    if poses.ndim == 4:
        poses = poses[:, 0]

    video = VideoReader(mp4_path)
    n_pose = len(poses)
    n_frames = min(n_pose, video.n_frames)
    if n_pose != video.n_frames:
        print(
            f"[warn] frames npy={n_pose} vs mp4={video.n_frames} → usando {n_frames}"
        )

    print(f"MP4: {mp4_path}  ({video.src_w}x{video.src_h})")
    print(f"Cargando detección… ({n_frames} frames @ {fps} fps)")
    cfg = DetectionConfig()
    segments, _ = ActionAnalysisPipeline(cfg).run(poses[:n_frames], fps)
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
            extra_header=[
                f"num_segments={len(segments)}",
                "from=validate_realtime_full",
                "config_variant=stable",
                f"mp4={mp4_path}",
            ],
        )
        print(f"TXT guardado: {out_path}")

    disp_w, disp_h = fit_size(video.src_w, video.src_h, VID_MAX_W, VID_MAX_H)
    delay0 = max(1, int(1000 / fps))
    state = {
        "frame": 0,
        "paused": False,
        "speed": 1.0,
        "only_code0": False,
        "filter": name_filter,
    }

    root = tk.Tk()
    root.title(f"Validación FULL — {npy_path.parent.name}/{npy_path.name}")
    root.resizable(False, False)
    label = tk.Label(root)
    label.pack()
    photo_ref = {"img": None}

    def refresh() -> None:
        f = state["frame"] % n_frames
        try:
            rgb = video.get_rgb(f)
        except Exception as e:
            print(f"[warn] frame {f}: {e}", file=sys.stderr)
            return
        active = active_segments_at_frame(segments, f)
        pose = poses[f] if f < n_pose else None
        img = draw_composite(
            rgb,
            pose,
            f,
            active,
            fps=fps,
            paused=state["paused"],
            speed=state["speed"],
            only_code0=state["only_code0"],
            name_filter=state["filter"],
            n_frames=n_frames,
            disp_w=disp_w,
            disp_h=disp_h,
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
            video.release()
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

    def on_close() -> None:
        video.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<KeyPress>", on_key)
    refresh()
    root.after(delay0, tick)
    root.mainloop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vídeo + esqueleto .npy + acciones detectadas in-situ"
    )
    p.add_argument(
        "npy",
        type=str,
        nargs="?",
        default=None,
        help=f"Ruta al .npy relativa a videos/ (default: {DEFAULT_NPY_RELATIVE})",
    )
    p.add_argument(
        "--mp4",
        type=str,
        default=None,
        help="Ruta al mp4 (default: <CLIP>/clip.mp4 junto al .npy)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS (si se omite, meta.json)",
    )
    p.add_argument("--save-txt", action="store_true", help="Guarda el .txt de acciones")
    p.add_argument(
        "--filter",
        type=str,
        default="",
        help="Filtrar acciones por nombre (ej: WRIST_NEAR)",
    )
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        npy = resolve_npy_path(args.npy)
    except Exception as e:
        print(f"Error resolviendo .npy: {e}", file=sys.stderr)
        return 2
    try:
        mp4 = resolve_clip_mp4(npy, args.mp4)
    except Exception as e:
        print(f"Error resolviendo mp4: {e}", file=sys.stderr)
        return 2
    try:
        fps, fps_src = resolve_fps(npy, args.fps)
    except Exception as e:
        print(f"Error resolviendo FPS: {e}", file=sys.stderr)
        return 2

    print(f"NPY: {npy}")
    print(f"MP4: {mp4}")
    print(f"FPS: {fps}  (origen: {fps_src})")

    out_dir = Path(args.output_dir) if args.output_dir else None
    run_viewer(
        npy,
        mp4,
        fps,
        save_txt=args.save_txt,
        name_filter=args.filter,
        output_dir=out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
