"""Extracción de features geométricas y dinámicas a partir de una secuencia de poses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .config import KEYPOINTS, DetectionConfig
from . import geometry as geo
from .filters import smooth_array, smooth_series, temporal_derivative


@dataclass
class PoseSequence:
    """
    Contenedor de poses 2D.

    Esperado: shape (T, J, 2) con J>=8 según KEYPOINTS.
    Extensible a (T, J, 3) ignorando z, o scores aparte.
    """

    poses: np.ndarray
    fps: float
    keypoints: Dict[str, int] = field(default_factory=lambda: dict(KEYPOINTS))
    confidence: np.ndarray | None = None  # opcional (T, J)

    def __post_init__(self) -> None:
        self.poses = np.asarray(self.poses, dtype=np.float64)
        if self.poses.ndim != 3 or self.poses.shape[-1] < 2:
            raise ValueError(
                f"Se espera shape (T, J, 2[+]), recibido {self.poses.shape}"
            )
        if self.poses.shape[-1] > 2:
            self.poses = self.poses[..., :2]
        max_idx = max(self.keypoints.values())
        if self.poses.shape[1] <= max_idx:
            raise ValueError(
                f"J={self.poses.shape[1]} insuficiente para keypoints {self.keypoints}"
            )
        if self.fps <= 0:
            raise ValueError(f"FPS inválido: {self.fps}")

    @property
    def num_frames(self) -> int:
        return self.poses.shape[0]

    def joint(self, name: str) -> np.ndarray:
        return self.poses[:, self.keypoints[name], :]

    @classmethod
    def from_npy(
        cls,
        path: str,
        fps: float,
        keypoints: Dict[str, int] | None = None,
    ) -> "PoseSequence":
        data = np.load(path, allow_pickle=False)
        if data.ndim == 4 and data.shape[1] == 2:
            # (T, 2 users, J, 2) → usar usuario 0
            data = data[:, 0]
        return cls(poses=data, fps=fps, keypoints=keypoints or dict(KEYPOINTS))


@dataclass
class FeatureSet:
    """Diccionario de series temporales (T,) + metadatos."""

    values: Dict[str, np.ndarray]
    fps: float
    num_frames: int
    body_scale: np.ndarray

    def get(self, name: str) -> np.ndarray:
        return self.values[name]


class FeatureExtractor:
    """Calcula features normalizadas y suavizadas frame a frame."""

    def __init__(self, config: DetectionConfig | None = None):
        self.cfg = config or DetectionConfig()

    def extract(self, seq: PoseSequence) -> FeatureSet:
        cfg = self.cfg
        T = seq.num_frames
        LS, RS = seq.joint("left_shoulder"), seq.joint("right_shoulder")
        LE, RE = seq.joint("left_elbow"), seq.joint("right_elbow")
        LW, RW = seq.joint("left_wrist"), seq.joint("right_wrist")
        LH, RH = seq.joint("left_hip"), seq.joint("right_hip")

        # Suavizar coordenadas antes de features
        coords = np.stack([LS, RS, LE, RE, LW, RW, LH, RH], axis=1)  # (T,8,2)
        coords = smooth_array(
            coords, cfg.smooth_window, cfg.smooth_method, cfg.savgol_polyorder
        )
        LS, RS, LE, RE, LW, RW, LH, RH = [coords[:, i] for i in range(8)]

        Sc = geo.midpoint(LS, RS)
        Hc = geo.midpoint(LH, RH)
        C = cfg.chest_alpha * Sc + (1.0 - cfg.chest_alpha) * Hc

        shoulder_w = geo.distance(LS, RS)
        torso_h = geo.distance(Sc, Hc)
        body_scale = (
            cfg.scale_shoulder_weight * shoulder_w + cfg.scale_torso_weight * torso_h
        )
        body_scale = np.maximum(body_scale, cfg.min_body_scale)
        # frames sin escala válida
        bad_scale = (~np.isfinite(shoulder_w)) & (~np.isfinite(torso_h))
        body_scale = np.where(bad_scale, np.nan, body_scale)

        def ndist(a, b):
            return geo.distance(a, b) / body_scale

        # Ángulos codo
        elbow_L = geo.angle_at_joint(LS, LE, LW)
        elbow_R = geo.angle_at_joint(RS, RE, RW)

        # Distancias normalizadas
        wrist_waist_L = ndist(LW, Hc)
        wrist_waist_R = ndist(RW, Hc)
        wrist_chest_L = ndist(LW, C)
        wrist_chest_R = ndist(RW, C)
        wrist_sh_L = ndist(LW, LS)
        wrist_sh_R = ndist(RW, RS)
        wrists_apart = ndist(LW, RW)

        # Alturas: positivo = muñeca más arriba (si y_down)
        sign_y = 1.0 if cfg.y_down else -1.0

        def rel_height(wrist, ref):
            return sign_y * (ref[..., 1] - wrist[..., 1]) / body_scale

        wrist_h_sh_L = rel_height(LW, LS)
        wrist_h_sh_R = rel_height(RW, RS)
        wrist_h_Sc_L = rel_height(LW, Sc)
        wrist_h_Sc_R = rel_height(RW, Sc)

        # Desplazamiento horizontal respecto a Sc (x_wrist - x_Sc) / scale
        lat_L = (LW[..., 0] - Sc[..., 0]) / body_scale
        lat_R = (RW[..., 0] - Sc[..., 0]) / body_scale
        # Cruce de línea media: L debería estar a la izq (x menor); cruce si lat_L > 0
        cross_L = lat_L  # >0 ⇒ muñeca izq a la derecha de la media
        cross_R = -lat_R  # >0 ⇒ muñeca der a la izquierda de la media

        # Apertura lateral respecto al hombro propio
        # Brazo izq abierto: wrist.x < shoulder.x (x crece a la derecha)
        open_L = (LS[..., 0] - LW[..., 0]) / body_scale
        open_R = (RW[..., 0] - RS[..., 0]) / body_scale
        open_both = 0.5 * (open_L + open_R)

        # Orientaciones
        shoulder_orient = geo.orientation_deg(LS, RS)
        hip_orient = geo.orientation_deg(LH, RH)
        torso_lean = geo.torso_lean_from_vertical_deg(Sc, Hc, cfg.y_down)
        twist = geo.angle_diff_deg(shoulder_orient, hip_orient)

        # Ratios heurísticos orientación
        frontal_ratio = shoulder_w / np.maximum(torso_h, cfg.min_body_scale)

        feats: Dict[str, np.ndarray] = {
            "elbow_angle_L": elbow_L,
            "elbow_angle_R": elbow_R,
            "wrist_waist_L": wrist_waist_L,
            "wrist_waist_R": wrist_waist_R,
            "wrist_chest_L": wrist_chest_L,
            "wrist_chest_R": wrist_chest_R,
            "wrist_shoulder_L": wrist_sh_L,
            "wrist_shoulder_R": wrist_sh_R,
            "wrists_apart": wrists_apart,
            "wrist_height_sh_L": wrist_h_sh_L,
            "wrist_height_sh_R": wrist_h_sh_R,
            "wrist_height_Sc_L": wrist_h_Sc_L,
            "wrist_height_Sc_R": wrist_h_Sc_R,
            "lateral_L": lat_L,
            "lateral_R": lat_R,
            "cross_L": cross_L,
            "cross_R": cross_R,
            "open_L": open_L,
            "open_R": open_R,
            "open_both": open_both,
            "shoulder_orient": shoulder_orient,
            "hip_orient": hip_orient,
            "torso_lean": torso_lean,
            "shoulder_hip_twist": twist,
            "shoulder_width": shoulder_w / body_scale,
            "torso_height": torso_h / body_scale,
            "frontal_ratio": frontal_ratio,
            "shoulder_c_x": Sc[..., 0],
            "shoulder_c_y": Sc[..., 1],
        }

        # Dinámica: suavizar features escalares clave y derivar
        dyn_keys = [
            "elbow_angle_L",
            "elbow_angle_R",
            "wrist_waist_L",
            "wrist_waist_R",
            "wrist_chest_L",
            "wrist_chest_R",
            "wrists_apart",
            "wrist_height_sh_L",
            "wrist_height_sh_R",
            "open_L",
            "open_R",
            "open_both",
            "shoulder_width",
            "frontal_ratio",
            "torso_lean",
        ]
        for k in dyn_keys:
            feats[k] = smooth_series(
                feats[k], cfg.smooth_window, cfg.smooth_method, cfg.savgol_polyorder
            )

        # Velocidades de muñecas/codos (desplazamiento / scale / s)
        def joint_speed(pts):
            # pts (T,2) ya suavizados en coords
            d = np.full(T, np.nan)
            if T < 2:
                return d
            delta = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
            d[1:] = delta * seq.fps / body_scale[1:]
            d[0] = d[1] if T > 1 else np.nan
            return smooth_series(d, cfg.velocity_smooth_window, "mean")

        feats["speed_wrist_L"] = joint_speed(LW)
        feats["speed_wrist_R"] = joint_speed(RW)
        feats["speed_elbow_L"] = joint_speed(LE)
        feats["speed_elbow_R"] = joint_speed(RE)

        for k in [
            "elbow_angle_L",
            "elbow_angle_R",
            "wrist_waist_L",
            "wrist_waist_R",
            "wrist_chest_L",
            "wrist_chest_R",
            "wrists_apart",
            "wrist_height_sh_L",
            "wrist_height_sh_R",
            "open_both",
            "open_L",
            "open_R",
            "shoulder_width",
        ]:
            feats[f"d_{k}"] = temporal_derivative(feats[k], seq.fps)
            feats[f"d_{k}"] = smooth_series(
                feats[f"d_{k}"], cfg.velocity_smooth_window, "mean"
            )

        return FeatureSet(
            values=feats, fps=seq.fps, num_frames=T, body_scale=body_scale
        )
