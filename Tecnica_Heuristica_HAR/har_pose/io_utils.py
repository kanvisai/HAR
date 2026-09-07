"""Carga/validación de .npy y escritura del .txt de resultados."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .config import (
    DEFAULT_NPY_RELATIVE,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    UNSUPPORTED_2D,
    VIDEOS_DIR,
    DetectionConfig,
    Segment,
)


def seconds_to_timestamp(seconds: float) -> str:
    """Convierte segundos a HH:MM:SS.mmm."""
    if not np.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def resolve_npy_path(npy_arg: str | Path | None = None) -> Path:
    """
    Resuelve la ruta al .npy respecto a videos/ del proyecto.

    Reglas:
      - None / vacío  →  videos/<DEFAULT_NPY_RELATIVE>
      - absoluta      →  tal cual
      - empieza por videos/ →  PROJECT_ROOT / ruta
      - relativa      →  VIDEOS_DIR / ruta
        ej: 1/CLIP/user_136/poses_full.npy
    """
    if npy_arg is None or str(npy_arg).strip() == "":
        path = VIDEOS_DIR / DEFAULT_NPY_RELATIVE
    else:
        path = Path(npy_arg).expanduser()
        if path.is_absolute():
            pass
        elif path.parts and path.parts[0] == "videos":
            path = PROJECT_ROOT / path
        else:
            path = VIDEOS_DIR / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el .npy: {path}\n"
            f"  VIDEOS_DIR = {VIDEOS_DIR}\n"
            f"  Pasa una ruta relativa a videos/ "
            f"(ej: 1/CLIP/user_136/poses_full.npy) o una ruta absoluta."
        )
    return path


def resolve_clip_mp4(npy_path: str | Path, mp4_arg: str | Path | None = None) -> Path:
    """
    Localiza clip.mp4 del mismo clip que el .npy.

    .../videos/1/CLIP/user_136/poses_full.npy  →  .../videos/1/CLIP/clip.mp4
    """
    if mp4_arg is not None and str(mp4_arg).strip():
        p = Path(mp4_arg).expanduser()
        if not p.is_absolute():
            if p.parts and p.parts[0] == "videos":
                p = PROJECT_ROOT / p
            else:
                p = VIDEOS_DIR / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"No existe el mp4: {p}")
        return p

    npy = Path(npy_path).resolve()
    parent = npy.parent
    clip_dir = parent.parent if parent.name.startswith("user_") else parent
    candidates = [
        clip_dir / "clip.mp4",
        clip_dir / "clip.avi",
        clip_dir / f"{clip_dir.name}.mp4",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        f"No se encontró clip.mp4 en {clip_dir}. "
        "Pasa --mp4 manualmente."
    )


def find_meta_json(npy_path: str | Path, max_levels: int = 4) -> Path | None:
    """
    Busca meta.json subiendo desde el .npy.

    Tipico: .../CLIP/user_183/poses_full.npy  →  .../CLIP/meta.json
    """
    path = Path(npy_path).resolve()
    cur = path.parent
    for _ in range(max_levels):
        candidate = cur / "meta.json"
        if candidate.is_file():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def load_fps_from_meta(npy_path: str | Path) -> float:
    """Lee el campo 'fps' del meta.json asociado al clip."""
    import json

    meta_path = find_meta_json(npy_path)
    if meta_path is None:
        raise FileNotFoundError(
            f"No se encontró meta.json cerca de {npy_path}. "
            "Pasa --fps manualmente."
        )
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if "fps" not in meta:
        raise KeyError(f"'fps' no está en {meta_path}")
    fps = float(meta["fps"])
    if fps <= 0:
        raise ValueError(f"fps inválido en {meta_path}: {fps}")
    return fps


def resolve_fps(npy_path: str | Path, fps_arg: float | None) -> tuple[float, str]:
    """
    Prioridad: --fps CLI > meta.json.
    Devuelve (fps, origen_descriptivo).
    """
    if fps_arg is not None:
        if fps_arg <= 0:
            raise ValueError(f"FPS inválido: {fps_arg}")
        return float(fps_arg), "cli"
    fps = load_fps_from_meta(npy_path)
    meta_path = find_meta_json(npy_path)
    return fps, str(meta_path)


def load_pose_npy(path: str | Path) -> np.ndarray:
    """
    Carga y valida un .npy de poses.

    Acepta (T, J, 2), (T, J, 3) → xy, o (T, 2, J, 2) → usuario 0.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    try:
        data = np.load(path, allow_pickle=False)
    except ValueError as e:
        raise ValueError(f"No se pudo cargar {path} (¿object array?): {e}") from e

    if data.ndim == 4 and data.shape[1] == 2 and data.shape[-1] >= 2:
        data = data[:, 0, :, :2]
    elif data.ndim == 3 and data.shape[-1] >= 2:
        data = data[..., :2]
    else:
        raise ValueError(
            f"Shape no soportado: {data.shape}. Esperado (T,J,2), (T,J,3) o (T,2,J,2)."
        )

    if data.shape[0] < 1 or data.shape[1] < 8:
        raise ValueError(f"Secuencia insuficiente: {data.shape}")
    return np.asarray(data, dtype=np.float64)


def source_run_dirname(npy_path: str | Path) -> str:
    """
    Subcarpeta de salida espejo del origen bajo videos/.

    .../videos/1/CLIP/user_136/poses.npy  →  1/CLIP
    .../CLIP/user_183/poses.npy           →  CLIP
    .../CLIP/poses.npy                    →  CLIP
    """
    path = Path(npy_path).resolve()
    parent = path.parent
    if parent.name.startswith("user_"):
        clip = parent.parent
        # Si el abuelo es una clase (0-7) bajo videos/, incluirla
        class_dir = clip.parent
        if class_dir.name.isdigit() or (
            VIDEOS_DIR.exists() and class_dir.parent.resolve() == VIDEOS_DIR.resolve()
        ):
            return str(Path(class_dir.name) / clip.name)
        return clip.name
    # .../videos/1/CLIP/poses.npy
    if parent.parent.name.isdigit() or (
        VIDEOS_DIR.exists() and parent.parent.resolve() == VIDEOS_DIR.resolve()
    ):
        return str(Path(parent.parent.name) / parent.name)
    return parent.name


def default_output_dir_for_npy(
    npy_path: str | Path, base_output_dir: Path | None = None
) -> Path:
    """output/<clase>/<CLIP>/  o  output/<CLIP>/"""
    base = Path(base_output_dir) if base_output_dir else DEFAULT_OUTPUT_DIR
    run_dir = base / source_run_dirname(npy_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def default_output_path(npy_path: str | Path, output_dir: Path | None = None) -> Path:
    """
    Por defecto:
      output/<clase>/<CLIP>/<user_XXX_>poses_full_actions.txt
    """
    npy_path = Path(npy_path)
    if output_dir is not None:
        out_root = Path(output_dir)
        run_dir = out_root / source_run_dirname(npy_path)
    else:
        run_dir = default_output_dir_for_npy(npy_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    stem = npy_path.stem
    parent = npy_path.parent.name
    name = (
        f"{parent}_{stem}_actions.txt"
        if parent.startswith("user_")
        else f"{stem}_actions.txt"
    )
    return run_dir / name


def active_segments_at_frame(segments: Sequence[Segment], frame: int) -> List[Segment]:
    """Segmentos activos en un frame (inclusivo)."""
    return [s for s in segments if s.start_frame <= frame <= s.end_frame]

def write_results_txt(
    path: str | Path,
    segments: Sequence[Segment],
    *,
    source: str,
    fps: float,
    config: DetectionConfig | None = None,
    extra_header: Iterable[str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        "# ACTION SEGMENTATION RESULTS",
        f"# source: {source}",
        f"# fps: {fps}",
        "# columns: ACTION | START | END | DURATION | CODE",
        "# CODE: 0=green(reliable 2D) 1=yellow(heuristic/perspective) 2=red(unsupported)",
        "#",
    ]
    if extra_header:
        for h in extra_header:
            lines.append(f"# {h}")
        lines.append("#")

    segs = sorted(segments, key=lambda s: (s.start_frame, s.name))
    for seg in segs:
        start = seconds_to_timestamp(seg.start_s(fps))
        end = seconds_to_timestamp(seg.end_s(fps))
        dur = f"{seg.duration_s(fps):.3f}"
        lines.append(f"{seg.name} | {start} | {end} | {dur} | {seg.code}")

    if not segs:
        lines.append("# (no segments detected)")

    lines.append("")
    lines.append("# UNSUPPORTED WITH 2D KEYPOINTS")
    for name in UNSUPPORTED_2D:
        lines.append(f"{name} | CODE=2")

    lines.append("")
    lines.append("# NOTE: 2D skeleton has no reliable depth. Yellow labels are")
    lines.append("# perspective heuristics, not true 3D measurements.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_debug_features(
    path: str | Path,
    features: dict,
    fps: float,
    keys: Sequence[str] | None = None,
) -> Path:
    """Guarda CSV simple de features para depuración."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(keys) if keys else sorted(features.keys())
    T = len(next(iter(features.values())))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", *keys])
        for i in range(T):
            row = [i, f"{i / fps:.4f}"]
            for k in keys:
                v = features[k][i]
                row.append(f"{v:.6f}" if np.isfinite(v) else "")
            w.writerow(row)
    return path
