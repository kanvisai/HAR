"""
Configuración centralizada del detector de poses.

IMPORTANTE: los thresholds son valores iniciales razonables para poses
normalizadas (coords ~[0,1]) con 8 keypoints upper-body. Deben calibrarse
con un dataset real antes de usar en producción.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Mapping de índices — NO asumir este orden fuera de esta configuración.
# ---------------------------------------------------------------------------
KEYPOINTS: Dict[str, int] = {
    "left_shoulder": 0,
    "right_shoulder": 1,
    "left_elbow": 2,
    "right_elbow": 3,
    "left_wrist": 4,
    "right_wrist": 5,
    "left_hip": 6,
    "right_hip": 7,
}

# Clases explícitamente no observables de forma fiable en 2D (código 2).
UNSUPPORTED_2D = (
    "ARM_FORWARD_DEPTH",
    "ARM_BACKWARD_DEPTH",
    "HAND_FRONT_BACK",
    "TORSO_ROTATION_3D_EXACT",
    "SHOULDER_ANGLE_3D",
    "WRIST_DEPTH",
)

DEFAULT_OUTPUT_DIR = Path(
    "/home/ignacio/Escritorio/Company/Tecnica_Heuristica_HAR/output"
)

# Raíz del proyecto y carpeta de datos (poses .npy)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = PROJECT_ROOT / "videos"

# Ejemplo por defecto bajo videos/
DEFAULT_NPY_RELATIVE = (
    "1/12Diciembre2025_12Diciembre2025_Cervezas_Cervezas_manana_002920_002950_1/"
    "user_136/poses_full.npy"
)


@dataclass
class DetectionConfig:
    """Thresholds y parámetros temporales. Documentados como calibrables."""

    # --- Geometría corporal ---
    chest_alpha: float = 0.65  # C = α·Sc + (1-α)·Hc
    # Escala: mezcla de ancho de hombros y altura torso (Sc→Hc)
    scale_shoulder_weight: float = 0.55
    scale_torso_weight: float = 0.45
    min_body_scale: float = 1e-3  # evita división por escala degenerada

    # --- Ángulos de codo (grados). Convención: 180° ≈ brazo recto ---
    # Calibrar: enter > exit para histeresis (extended / straight)
    elbow_extended_enter_deg: float = 155.0
    elbow_extended_exit_deg: float = 140.0
    elbow_straight_enter_deg: float = 165.0
    elbow_straight_exit_deg: float = 155.0
    elbow_flexed_enter_deg: float = 95.0
    elbow_flexed_exit_deg: float = 110.0
    # "recogido": codo flexionado + muñeca cerca del torso
    arm_tucked_wrist_chest_norm: float = 0.55

    # --- Distancias normalizadas (unidades de escala corporal) ---
    wrist_waist_enter_norm: float = 0.45
    wrist_waist_exit_norm: float = 0.60
    wrist_chest_enter_norm: float = 0.40
    wrist_chest_exit_norm: float = 0.55
    wrist_shoulder_enter_norm: float = 0.35
    wrist_shoulder_exit_norm: float = 0.50
    wrists_together_enter_norm: float = 0.35
    wrists_together_exit_norm: float = 0.50
    wrists_apart_enter_norm: float = 1.10
    wrists_apart_exit_norm: float = 0.90

    # --- Alturas relativas (y_down: menor y = más arriba en imagen típica) ---
    # Usamos feature wrist_above_shoulder = (shoulder_y - wrist_y) / scale
    # Positivo => muñeca por encima del hombro (si y crece hacia abajo).
    wrist_above_shoulder_enter: float = 0.15
    wrist_above_shoulder_exit: float = 0.05
    arm_elevated_enter: float = 0.25
    arm_elevated_exit: float = 0.10
    arm_down_enter: float = -0.35  # muñeca claramente por debajo del hombro
    arm_down_exit: float = -0.20
    arm_horizontal_enter_abs: float = 0.18  # |altura rel.| pequeña
    arm_horizontal_exit_abs: float = 0.28

    # --- Apertura lateral (muñeca vs hombro en eje x, normalizado) ---
    # lateral_open = (wrist_x - shoulder_x) * side_sign / scale  (side_sign: L=-1,R=+1
    # en coords x-right: brazo izq abierto => wrist a la izquierda del hombro)
    arms_open_enter: float = 0.55
    arms_open_exit: float = 0.35

    # --- Inclinación torso (ángulo del eje Sc→Hc respecto a vertical) ---
    torso_lean_enter_deg: float = 12.0
    torso_lean_exit_deg: float = 7.0

    # --- Cruce de manos / brazos cruzados (heurística 2D → código 1) ---
    # HEURÍSTICO: en perfil o foreshortening el "cruce" 2D no implica cruce 3D.
    cross_midline_enter: float = 0.12  # muñeca supera línea media Sc
    cross_midline_exit: float = 0.05
    arms_crossed_require_both: bool = True

    # --- Heurísticas de orientación (código 1) ---
    # HEURÍSTICO: ratio ancho_hombros / altura_torso baja ⇒ posible perfil.
    frontal_ratio_enter: float = 0.55  # por encima ⇒ frontal aproximado
    frontal_ratio_exit: float = 0.40
    profile_ratio_enter: float = 0.28  # por debajo ⇒ perfil aproximado
    profile_ratio_exit: float = 0.38
    # Diferencia angular hombros vs cintura (grados) — torsión aparente
    shoulder_hip_twist_enter_deg: float = 18.0
    shoulder_hip_twist_exit_deg: float = 10.0
    # Rotación parcial: cambio temporal de ancho de hombros (norm.)
    shoulder_width_change_enter: float = 0.25
    # Giro izq/der: signo del lean + cambio de ancho (muy heurístico)
    body_turn_lean_enter_deg: float = 8.0

    # --- Acciones dinámicas (máquinas de estados) ---
    # Incremento mínimo de ángulo de codo para EXTEND_ARM (grados)
    extend_delta_elbow_deg: float = 25.0
    retract_delta_elbow_deg: float = 25.0
    # Reducción mínima de distancia para WRIST_TO_* (norm)
    approach_delta_norm: float = 0.25
    # Velocidad mínima (norm / s) para considerar movimiento significativo
    min_action_speed_norm_s: float = 0.15
    # Separar / juntar manos
    hands_sep_delta_norm: float = 0.30
    # Elevar / bajar: cambio de altura relativa
    raise_delta_height: float = 0.30
    # Abrir / cerrar brazos lateralmente
    open_close_delta: float = 0.30

    # --- Suavizado ---
    smooth_window: int = 5  # impar preferible
    smooth_method: str = "savgol"  # "savgol" | "mean" | "median"
    savgol_polyorder: int = 2
    velocity_smooth_window: int = 5

    # --- Temporal / segmentos ---
    min_state_duration_s: float = 0.20
    min_action_duration_s: float = 0.12
    merge_gap_s: float = 0.12
    # Frames mínimos de confirmación en FSM de acciones
    action_min_frames: int = 3

    # --- Imagen: y crece hacia abajo (OpenCV / vídeo típico) ---
    y_down: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Segment:
    """Segmento temporal de una acción/estado."""

    name: str
    start_frame: int
    end_frame: int  # inclusivo
    code: int  # 0 verde, 1 amarillo, 2 rojo
    reason: str = ""
    feature: str = ""
    enter_threshold: float | None = None
    exit_threshold: float | None = None

    def duration_s(self, fps: float) -> float:
        return (self.end_frame - self.start_frame + 1) / fps

    def start_s(self, fps: float) -> float:
        return self.start_frame / fps

    def end_s(self, fps: float) -> float:
        return (self.end_frame + 1) / fps
