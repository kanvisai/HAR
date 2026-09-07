"""Suavizado temporal de series con NaN."""
from __future__ import annotations

import numpy as np

try:
    from scipy.signal import savgol_filter

    _HAS_SAVGOL = True
except Exception:  # pragma: no cover
    _HAS_SAVGOL = False


def _odd_window(n: int, max_len: int) -> int:
    w = max(3, int(n))
    if w % 2 == 0:
        w += 1
    if w > max_len:
        w = max_len if max_len % 2 == 1 else max(1, max_len - 1)
    return max(1, w)


def moving_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Media móvil 1D ignorando NaN (ventana centrada aproximada causal+acausal)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = _odd_window(window, n)
    half = w // 2
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = x[lo:hi]
        valid = seg[np.isfinite(seg)]
        if valid.size:
            out[i] = valid.mean()
    return out


def moving_median(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = _odd_window(window, n)
    half = w // 2
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = x[lo:hi]
        valid = seg[np.isfinite(seg)]
        if valid.size:
            out[i] = np.median(valid)
    return out


def smooth_series(
    x: np.ndarray,
    window: int = 5,
    method: str = "savgol",
    polyorder: int = 2,
) -> np.ndarray:
    """
    Suaviza una serie 1D. Interpola NaNs linealmente para filtros globales
    y restaura NaNs donde la serie original era inválida en tramos largos.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("smooth_series espera array 1D")
    n = len(x)
    if n == 0:
        return x.copy()

    valid = np.isfinite(x)
    if valid.sum() < 3:
        return x.copy()

    # Interpolación lineal de gaps cortos para poder filtrar
    idx = np.arange(n)
    filled = x.copy()
    filled[~valid] = np.interp(idx[~valid], idx[valid], x[valid])

    method = (method or "mean").lower()
    w = _odd_window(window, n)

    if method == "median":
        out = moving_median(filled, w)
    elif method == "savgol" and _HAS_SAVGOL and w >= polyorder + 2:
        out = savgol_filter(filled, window_length=w, polyorder=min(polyorder, w - 1))
    else:
        out = moving_mean(filled, w)

    # Mantener NaN en extremos sin soporte o gaps muy largos (> window)
    # Marcar gaps largos
    out = out.astype(float)
    gap_mask = ~valid
    # dilatar un poco: si el frame original era NaN, conservar NaN
    # (evita inventar poses en frames inválidos)
    out[gap_mask] = np.nan
    return out


def smooth_array(
    arr: np.ndarray,
    window: int = 5,
    method: str = "savgol",
    polyorder: int = 2,
) -> np.ndarray:
    """Suaviza el último eje independiente... o cada columna si 2D (T, F)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return smooth_series(arr, window, method, polyorder)
    out = np.empty_like(arr, dtype=float)
    flat = arr.reshape(arr.shape[0], -1)
    out_flat = np.empty_like(flat)
    for j in range(flat.shape[1]):
        out_flat[:, j] = smooth_series(flat[:, j], window, method, polyorder)
    return out_flat.reshape(arr.shape)


def temporal_derivative(x: np.ndarray, fps: float) -> np.ndarray:
    """Derivada temporal ≈ dx/dt con diferencias centrales; NaN-safe."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if n < 2 or fps <= 0:
        return out
    dt = 1.0 / fps
    # forward/backward en extremos
    if n >= 2:
        out[0] = (x[1] - x[0]) / dt
        out[-1] = (x[-1] - x[-2]) / dt
    for i in range(1, n - 1):
        if np.isfinite(x[i + 1]) and np.isfinite(x[i - 1]):
            out[i] = (x[i + 1] - x[i - 1]) / (2 * dt)
        elif np.isfinite(x[i + 1]) and np.isfinite(x[i]):
            out[i] = (x[i + 1] - x[i]) / dt
        elif np.isfinite(x[i]) and np.isfinite(x[i - 1]):
            out[i] = (x[i] - x[i - 1]) / dt
    # extremos con NaN si faltan vecinos
    if not (np.isfinite(x[0]) and np.isfinite(x[1])):
        out[0] = np.nan
    if not (np.isfinite(x[-1]) and np.isfinite(x[-2])):
        out[-1] = np.nan
    return out
