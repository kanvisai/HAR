"""Carga de poses_full.npy y geometría del esqueleto de 8 puntos."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Orden fijo del .npy (YOLO pose, parte superior del cuerpo).
NOMBRES = (
    "hombro_izq",
    "hombro_der",
    "codo_izq",
    "codo_der",
    "muneca_izq",
    "muneca_der",
    "cadera_izq",
    "cadera_der",
)
HOMBRO_I, HOMBRO_D = 0, 1
CODO_I, CODO_D = 2, 3
MUNECA_I, MUNECA_D = 4, 5
CADERA_I, CADERA_D = 6, 7

HUESOS = (
    (HOMBRO_I, HOMBRO_D),
    (HOMBRO_I, CODO_I),
    (CODO_I, MUNECA_I),
    (HOMBRO_D, CODO_D),
    (CODO_D, MUNECA_D),
    (HOMBRO_I, CADERA_I),
    (HOMBRO_D, CADERA_D),
    (CADERA_I, CADERA_D),
)

LADO_BRAZO = {
    "izquierdo": (HOMBRO_I, CODO_I, MUNECA_I, CADERA_I),
    "derecho": (HOMBRO_D, CODO_D, MUNECA_D, CADERA_D),
}

EJEMPLO_POSES = Path(
    "/home/ignacio/Escritorio/Company/HAR/videos/6/"
    "robos_3_clips_clip61_000000_000020_6/user_15361/poses_full.npy"
)
EJEMPLO_VIDEO = Path(
    "/home/ignacio/Escritorio/Company/HAR/videos/6/"
    "robos_3_clips_clip61_000000_000020_6/clip.mp4"
)


def cargar_poses(path: Path) -> np.ndarray:
    """Devuelve (frames, 8, 2) en coordenadas normalizadas, y hacia abajo."""
    poses = np.load(path).astype(np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (8, 2):
        raise ValueError(
            f"{path} tiene shape {poses.shape}. Se espera (frames, 8, 2)."
        )
    return poses


def fps_y_video(poses_path: Path, video: Path | None = None) -> tuple[float, Path | None]:
    """Lee fps de meta.json del clip y localiza clip.mp4 si no se pasa."""
    clip_dir = poses_path.parent.parent if poses_path.parent.name.startswith("user_") else poses_path.parent
    meta_path = clip_dir / "meta.json"
    fps = 12.5
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fps = float(meta.get("fps", fps))
        if video is None:
            nombre = meta.get("clip_video", "clip.mp4")
            candidato = clip_dir / nombre
            if candidato.is_file():
                video = candidato
    if video is None:
        candidato = clip_dir / "clip.mp4"
        if candidato.is_file():
            video = candidato
    return fps, video


def suavizar(poses: np.ndarray, ventana: int = 5) -> np.ndarray:
    """Mediana temporal. No rellena huecos: un NaN en la ventana deja NaN."""
    if ventana < 3 or ventana % 2 == 0:
        raise ValueError("ventana debe ser impar y >= 3")
    mitad = ventana // 2
    relleno = np.pad(poses, ((mitad, mitad), (0, 0), (0, 0)), mode="edge")
    ventanas = np.lib.stride_tricks.sliding_window_view(relleno, ventana, axis=0)
    return np.median(ventanas, axis=-1)


def formatear_tiempo(segundos: float) -> str:
    segundos = max(0.0, segundos)
    minutos = int(segundos // 60)
    resto = segundos - minutos * 60
    return f"{minutos:02d}:{resto:05.2f}"
