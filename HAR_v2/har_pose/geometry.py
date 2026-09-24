"""Operaciones geométricas 2D vectorizadas y seguras ante NaN/degeneración."""
from __future__ import annotations

import numpy as np


def safe_norm(v: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Norma L2 con suelo eps para evitar división por cero."""
    return np.maximum(np.linalg.norm(v, axis=axis), eps)


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Punto medio; propaga NaN si alguno es inválido."""
    return 0.5 * (a + b)


def angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Ángulo en B del triángulo A-B-C, en grados, rango [0, 180].

    a, b, c: (..., 2)
    Si vectores degenerados o NaN → NaN.
    """
    ba = a - b
    bc = c - b
    ba_n = safe_norm(ba)
    bc_n = safe_norm(bc)
    # Detectar degeneración real (norma casi 0 antes del eps)
    ba_raw = np.linalg.norm(ba, axis=-1)
    bc_raw = np.linalg.norm(bc, axis=-1)
    cos = np.sum(ba * bc, axis=-1) / (ba_n * bc_n)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    invalid = (~np.isfinite(a).all(axis=-1)) | (~np.isfinite(b).all(axis=-1)) | (
        ~np.isfinite(c).all(axis=-1)
    )
    invalid |= (ba_raw < 1e-8) | (bc_raw < 1e-8)
    ang = np.where(invalid, np.nan, ang)
    return ang


def orientation_deg(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Ángulo del vector p0→p1 respecto al eje +x, en grados [-180, 180]."""
    d = p1 - p0
    ang = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
    invalid = (~np.isfinite(p0).all(axis=-1)) | (~np.isfinite(p1).all(axis=-1))
    return np.where(invalid, np.nan, ang)


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Diferencia angular firmada en [-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def torso_lean_from_vertical_deg(
    shoulder_c: np.ndarray, hip_c: np.ndarray, y_down: bool = True
) -> np.ndarray:
    """
    Inclinación del torso (Sc→Hc) respecto a la vertical.

    Positivo = inclinación hacia la derecha de la imagen (x creciente).
    """
    d = hip_c - shoulder_c
    # Vertical de referencia: (0, +1) si y_down, else (0, -1)
    # Ángulo respecto a vertical: atan2(dx, dy_aligned)
    dy = d[..., 1] if y_down else -d[..., 1]
    ang = np.degrees(np.arctan2(d[..., 0], dy))
    invalid = (~np.isfinite(shoulder_c).all(axis=-1)) | (~np.isfinite(hip_c).all(axis=-1))
    return np.where(invalid, np.nan, ang)


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distancia euclídea; NaN si algún punto inválido."""
    d = np.linalg.norm(a - b, axis=-1)
    invalid = (~np.isfinite(a).all(axis=-1)) | (~np.isfinite(b).all(axis=-1))
    return np.where(invalid, np.nan, d)
