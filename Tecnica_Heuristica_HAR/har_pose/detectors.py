"""
Detectores de estados (persistentes) y acciones (transiciones dinámicas).

Código 0: reglas 2D geométricas robustas.
Código 1: heurísticas dependientes de perspectiva (comentadas como tales).
Código 2: nunca se detectan aquí; solo se listan en la salida.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import DetectionConfig, Segment
from .features import FeatureSet
from .temporal import SegmentProcessor, hysteresis_mask, mask_to_segments


@dataclass
class DetectorMeta:
    """Metadatos de explicabilidad de un detector."""

    name: str
    feature: str
    enter: float
    exit: float
    mode: str
    code: int
    reason: str


class StateDetector:
    """Detecta estados corporales con histeresis sobre features."""

    def __init__(self, config: DetectionConfig, fps: float):
        self.cfg = config
        self.fps = fps
        self.proc = SegmentProcessor(config, fps)
        self.last_metas: List[DetectorMeta] = []

    def _run(
        self,
        feats: FeatureSet,
        name: str,
        feature: str,
        enter: float,
        exit: float,
        mode: str,
        code: int,
        reason: str,
    ) -> List[Segment]:
        values = feats.get(feature)
        mask = hysteresis_mask(values, enter, exit, mode=mode)
        self.last_metas.append(
            DetectorMeta(name, feature, enter, exit, mode, code, reason)
        )
        segs = mask_to_segments(
            mask,
            name,
            code,
            reason=reason,
            feature=feature,
            enter_threshold=enter,
            exit_threshold=exit,
        )
        return self.proc.process(segs, action=False)

    def detect_all(self, feats: FeatureSet) -> List[Segment]:
        c = self.cfg
        out: List[Segment] = []
        self.last_metas = []

        # --- Extensión / flexión / casi recto (ángulos 2D fiables → código 0) ---
        for side, letter in (("LEFT", "L"), ("RIGHT", "R")):
            out += self._run(
                feats,
                f"ARM_EXTENDED_{side}",
                f"elbow_angle_{letter}",
                c.elbow_extended_enter_deg,
                c.elbow_extended_exit_deg,
                "above",
                0,
                "elbow angle above extended threshold (2D)",
            )
            out += self._run(
                feats,
                f"ARM_FLEXED_{side}",
                f"elbow_angle_{letter}",
                c.elbow_flexed_enter_deg,
                c.elbow_flexed_exit_deg,
                "below",
                0,
                "elbow angle below flexed threshold (2D)",
            )
            out += self._run(
                feats,
                f"ELBOW_ALMOST_STRAIGHT_{side}",
                f"elbow_angle_{letter}",
                c.elbow_straight_enter_deg,
                c.elbow_straight_exit_deg,
                "above",
                0,
                "elbow nearly straight (2D)",
            )
            # Recogido: flexionado + muñeca cerca del pecho (combinación)
            flexed = hysteresis_mask(
                feats.get(f"elbow_angle_{letter}"),
                c.elbow_flexed_enter_deg,
                c.elbow_flexed_exit_deg,
                mode="below",
            )
            near_chest = hysteresis_mask(
                feats.get(f"wrist_chest_{letter}"),
                c.arm_tucked_wrist_chest_norm,
                c.wrist_chest_exit_norm,
                mode="below",
            )
            tucked = flexed & near_chest
            segs = mask_to_segments(
                tucked,
                f"ARM_TUCKED_{side}",
                0,
                reason="flexed elbow + wrist near chest",
                feature=f"elbow_angle_{letter}&wrist_chest_{letter}",
            )
            out += self.proc.process(segs)

            # Proximidades
            # Proximidad muñeca → cintura / pecho (CODE=0, distancia 2D normalizada)
            near_waist_mask = hysteresis_mask(
                feats.get(f"wrist_waist_{letter}"),
                c.wrist_waist_enter_norm,
                c.wrist_waist_exit_norm,
                mode="below",
            )
            near_chest_mask = hysteresis_mask(
                feats.get(f"wrist_chest_{letter}"),
                c.wrist_chest_enter_norm,
                c.wrist_chest_exit_norm,
                mode="below",
            )
            out += self.proc.process(
                mask_to_segments(
                    near_waist_mask,
                    f"WRIST_NEAR_WAIST_{side}",
                    0,
                    reason="muñeca cerca de la cintura (distancia normalizada)",
                    feature=f"wrist_waist_{letter}",
                    enter_threshold=c.wrist_waist_enter_norm,
                    exit_threshold=c.wrist_waist_exit_norm,
                )
            )
            out += self.proc.process(
                mask_to_segments(
                    near_chest_mask,
                    f"WRIST_NEAR_CHEST_{side}",
                    0,
                    reason="muñeca cerca del pecho (distancia normalizada)",
                    feature=f"wrist_chest_{letter}",
                    enter_threshold=c.wrist_chest_enter_norm,
                    exit_threshold=c.wrist_chest_exit_norm,
                )
            )
            # Cerca del torso: pecho O cintura
            out += self.proc.process(
                mask_to_segments(
                    near_waist_mask | near_chest_mask,
                    f"WRIST_NEAR_TORSO_{side}",
                    0,
                    reason="muñeca cerca de pecho o cintura",
                    feature=f"wrist_chest_{letter}|wrist_waist_{letter}",
                )
            )
            if side == "LEFT":
                near_waist_L, near_chest_L = near_waist_mask, near_chest_mask
            else:
                near_waist_R, near_chest_R = near_waist_mask, near_chest_mask

            out += self._run(
                feats,
                f"WRIST_NEAR_SHOULDER_{side}",
                f"wrist_shoulder_{letter}",
                c.wrist_shoulder_enter_norm,
                c.wrist_shoulder_exit_norm,
                "below",
                0,
                "normalized wrist-shoulder distance",
            )

            # Elevación / abajo / horizontal
            out += self._run(
                feats,
                f"ARM_ELEVATED_{side}",
                f"wrist_height_sh_{letter}",
                c.arm_elevated_enter,
                c.arm_elevated_exit,
                "above",
                0,
                "wrist above shoulder (image y)",
            )
            out += self._run(
                feats,
                f"ARM_DOWN_{side}",
                f"wrist_height_sh_{letter}",
                c.arm_down_enter,
                c.arm_down_exit,
                "below",
                0,
                "wrist below shoulder (image y)",
            )
            out += self._run(
                feats,
                f"ARM_HORIZONTAL_{side}",
                f"wrist_height_sh_{letter}",
                c.arm_horizontal_enter_abs,
                c.arm_horizontal_exit_abs,
                "abs_below",
                0,
                "wrist roughly shoulder height",
            )

        # --- Ambas muñecas cerca de pecho / cintura ---
        out += self.proc.process(
            mask_to_segments(
                near_chest_L & near_chest_R,
                "BOTH_WRISTS_NEAR_CHEST",
                0,
                reason="ambas muñecas cerca del pecho",
                feature="wrist_chest_L&wrist_chest_R",
                enter_threshold=c.wrist_chest_enter_norm,
                exit_threshold=c.wrist_chest_exit_norm,
            )
        )
        out += self.proc.process(
            mask_to_segments(
                near_waist_L & near_waist_R,
                "BOTH_WRISTS_NEAR_WAIST",
                0,
                reason="ambas muñecas cerca de la cintura",
                feature="wrist_waist_L&wrist_waist_R",
                enter_threshold=c.wrist_waist_enter_norm,
                exit_threshold=c.wrist_waist_exit_norm,
            )
        )
        # Al menos una muñeca cerca del pecho / de la cintura
        out += self.proc.process(
            mask_to_segments(
                near_chest_L | near_chest_R,
                "ANY_WRIST_NEAR_CHEST",
                0,
                reason="al menos una muñeca cerca del pecho",
                feature="wrist_chest_L|wrist_chest_R",
            )
        )
        out += self.proc.process(
            mask_to_segments(
                near_waist_L | near_waist_R,
                "ANY_WRIST_NEAR_WAIST",
                0,
                reason="al menos una muñeca cerca de la cintura",
                feature="wrist_waist_L|wrist_waist_R",
            )
        )

        # Muñecas juntas / separadas
        out += self._run(
            feats,
            "WRISTS_TOGETHER",
            "wrists_apart",
            c.wrists_together_enter_norm,
            c.wrists_together_exit_norm,
            "below",
            0,
            "both wrists close in 2D",
        )
        out += self._run(
            feats,
            "WRISTS_APART",
            "wrists_apart",
            c.wrists_apart_enter_norm,
            c.wrists_apart_exit_norm,
            "above",
            0,
            "both wrists far in 2D",
        )

        # Apertura lateral de brazos (estado)
        out += self._run(
            feats,
            "ARMS_OPEN_LATERAL",
            "open_both",
            c.arms_open_enter,
            c.arms_open_exit,
            "above",
            0,
            "both arms laterally open vs shoulders",
        )

        # Muñecas por encima de hombros (ambas)
        above_L = hysteresis_mask(
            feats.get("wrist_height_sh_L"),
            c.wrist_above_shoulder_enter,
            c.wrist_above_shoulder_exit,
            mode="above",
        )
        above_R = hysteresis_mask(
            feats.get("wrist_height_sh_R"),
            c.wrist_above_shoulder_enter,
            c.wrist_above_shoulder_exit,
            mode="above",
        )
        both_above = above_L & above_R
        out += self.proc.process(
            mask_to_segments(
                both_above,
                "WRISTS_ABOVE_SHOULDERS",
                0,
                reason="both wrists above shoulders",
                feature="wrist_height_sh_L&R",
            )
        )

        # Ambos brazos elevados
        elev_L = hysteresis_mask(
            feats.get("wrist_height_sh_L"),
            c.arm_elevated_enter,
            c.arm_elevated_exit,
            mode="above",
        )
        elev_R = hysteresis_mask(
            feats.get("wrist_height_sh_R"),
            c.arm_elevated_enter,
            c.arm_elevated_exit,
            mode="above",
        )
        out += self.proc.process(
            mask_to_segments(
                elev_L & elev_R,
                "BOTH_ARMS_ELEVATED",
                0,
                reason="both arms elevated",
                feature="wrist_height_sh_L&R",
            )
        )

        # Inclinar torso (ángulo 2D del eje Sc-Hc → código 0 razonable en frontal)
        lean = feats.get("torso_lean")
        out += self.proc.process(
            mask_to_segments(
                hysteresis_mask(lean, c.torso_lean_enter_deg, c.torso_lean_exit_deg, mode="above"),
                "TORSO_LEAN_RIGHT",
                0,
                reason="torso axis leans right in image",
                feature="torso_lean",
                enter_threshold=c.torso_lean_enter_deg,
                exit_threshold=c.torso_lean_exit_deg,
            )
        )
        out += self.proc.process(
            mask_to_segments(
                hysteresis_mask(lean, -c.torso_lean_enter_deg, -c.torso_lean_exit_deg, mode="below"),
                "TORSO_LEAN_LEFT",
                0,
                reason="torso axis leans left in image",
                feature="torso_lean",
                enter_threshold=-c.torso_lean_enter_deg,
                exit_threshold=-c.torso_lean_exit_deg,
            )
        )

        # --- Código 1: cruce / brazos cruzados (perspectiva) ---
        # HEURÍSTICO: cruce de línea media en 2D no implica cruce anatómico 3D
        # (perfil, foreshortening, oclusión). Código 1.
        for side, letter in (("LEFT", "L"), ("RIGHT", "R")):
            out += self._run(
                feats,
                f"HAND_CROSS_MIDLINE_{side}",
                f"cross_{letter}",
                c.cross_midline_enter,
                c.cross_midline_exit,
                "above",
                1,
                "HEURISTIC: 2D midline crossing depends on camera view",
            )

        cross_L = hysteresis_mask(
            feats.get("cross_L"), c.cross_midline_enter, c.cross_midline_exit, mode="above"
        )
        cross_R = hysteresis_mask(
            feats.get("cross_R"), c.cross_midline_enter, c.cross_midline_exit, mode="above"
        )
        crossed = (cross_L & cross_R) if c.arms_crossed_require_both else (cross_L | cross_R)
        out += self.proc.process(
            mask_to_segments(
                crossed,
                "ARMS_CROSSED",
                1,
                reason="HEURISTIC: both wrists past midline in 2D",
                feature="cross_L&cross_R",
            )
        )

        # --- Heurísticas de orientación (código 1) ---
        # HEURÍSTICO: ratio ancho_hombros/altura_torso ≈ frontalidad aparente.
        out += self._run(
            feats,
            "PERSON_FRONTAL_HEURISTIC",
            "frontal_ratio",
            c.frontal_ratio_enter,
            c.frontal_ratio_exit,
            "above",
            1,
            "HEURISTIC: high shoulder-width/torso-height ratio ≈ frontal view",
        )
        out += self._run(
            feats,
            "PERSON_PROFILE_HEURISTIC",
            "frontal_ratio",
            c.profile_ratio_enter,
            c.profile_ratio_exit,
            "below",
            1,
            "HEURISTIC: low shoulder-width/torso-height ratio ≈ profile view",
        )
        # Torsión aparente hombros vs cintura
        twist = np.abs(feats.get("shoulder_hip_twist"))
        out += self.proc.process(
            mask_to_segments(
                hysteresis_mask(
                    twist,
                    c.shoulder_hip_twist_enter_deg,
                    c.shoulder_hip_twist_exit_deg,
                    mode="above",
                ),
                "SHOULDER_HIP_TWIST_HEURISTIC",
                1,
                reason="HEURISTIC: 2D angle diff shoulders vs hips ≠ true 3D torsion",
                feature="shoulder_hip_twist",
                enter_threshold=c.shoulder_hip_twist_enter_deg,
                exit_threshold=c.shoulder_hip_twist_exit_deg,
            )
        )

        # Rotación parcial: |d(shoulder_width)/dt| alto
        dsw = np.abs(feats.get("d_shoulder_width"))
        # Convertir threshold de cambio acumulado a velocidad aproximada:
        # enter si |d_sw| > shoulder_width_change_enter * fps/4 (heurístico)
        rot_enter = c.shoulder_width_change_enter * 0.5
        rot_exit = rot_enter * 0.5
        out += self.proc.process(
            mask_to_segments(
                hysteresis_mask(dsw, rot_enter, rot_exit, mode="above"),
                "PARTIAL_TORSO_ROTATION_HEURISTIC",
                1,
                reason="HEURISTIC: apparent shoulder-width change over time",
                feature="d_shoulder_width",
                enter_threshold=rot_enter,
                exit_threshold=rot_exit,
            )
        )

        # Giro izq/der del cuerpo: lean + cambio de ancho (muy heurístico)
        lean_v = feats.get("torso_lean")
        turning = hysteresis_mask(dsw, rot_enter, rot_exit, mode="above")
        turn_L = turning & hysteresis_mask(
            lean_v, -c.body_turn_lean_enter_deg, -c.body_turn_lean_enter_deg * 0.5, mode="below"
        )
        turn_R = turning & hysteresis_mask(
            lean_v, c.body_turn_lean_enter_deg, c.body_turn_lean_enter_deg * 0.5, mode="above"
        )
        out += self.proc.process(
            mask_to_segments(
                turn_L,
                "BODY_TURN_LEFT_HEURISTIC",
                1,
                reason="HEURISTIC: lean left + shoulder-width dynamics",
                feature="torso_lean&d_shoulder_width",
            )
        )
        out += self.proc.process(
            mask_to_segments(
                turn_R,
                "BODY_TURN_RIGHT_HEURISTIC",
                1,
                reason="HEURISTIC: lean right + shoulder-width dynamics",
                feature="torso_lean&d_shoulder_width",
            )
        )

        return out


class ActionDetector:
    """
    Acciones dinámicas mediante máquinas de estados temporales.

    No etiqueta una acción solo porque el estado final ya esté presente
    al inicio del vídeo: exige transición observable.
    """

    def __init__(self, config: DetectionConfig, fps: float):
        self.cfg = config
        self.fps = fps
        self.proc = SegmentProcessor(config, fps)

    def detect_all(self, feats: FeatureSet) -> List[Segment]:
        c = self.cfg
        out: List[Segment] = []

        for side, letter in (("LEFT", "L"), ("RIGHT", "R")):
            out += self._transition_increase(
                feats,
                name=f"EXTEND_ARM_{side}",
                value_key=f"elbow_angle_{letter}",
                deriv_key=f"d_elbow_angle_{letter}",
                delta=c.extend_delta_elbow_deg,
                end_enter=c.elbow_extended_enter_deg,
                end_exit=c.elbow_extended_exit_deg,
                start_below=c.elbow_extended_exit_deg,
                code=0,
                reason="elbow angle increases into extended state",
            )
            out += self._transition_decrease(
                feats,
                name=f"RETRACT_ARM_{side}",
                value_key=f"elbow_angle_{letter}",
                deriv_key=f"d_elbow_angle_{letter}",
                delta=c.retract_delta_elbow_deg,
                end_enter=c.elbow_flexed_enter_deg,
                end_exit=c.elbow_flexed_exit_deg,
                start_above=c.elbow_flexed_exit_deg,
                end_mode="below",
                code=0,
                reason="elbow angle decreases into flexed state",
            )
            out += self._approach(
                feats,
                name=f"WRIST_TO_WAIST_{side}",
                dist_key=f"wrist_waist_{letter}",
                deriv_key=f"d_wrist_waist_{letter}",
                speed_key=f"speed_wrist_{letter}",
                near_enter=c.wrist_waist_enter_norm,
                near_exit=c.wrist_waist_exit_norm,
                code=0,
                reason="wrist-waist distance decreases into proximity",
            )
            out += self._approach(
                feats,
                name=f"WRIST_TO_CHEST_{side}",
                dist_key=f"wrist_chest_{letter}",
                deriv_key=f"d_wrist_chest_{letter}",
                speed_key=f"speed_wrist_{letter}",
                near_enter=c.wrist_chest_enter_norm,
                near_exit=c.wrist_chest_exit_norm,
                code=0,
                reason="wrist-chest distance decreases into proximity",
            )
            out += self._height_change(
                feats,
                name=f"RAISE_ARM_{side}",
                height_key=f"wrist_height_sh_{letter}",
                deriv_key=f"d_wrist_height_sh_{letter}",
                delta=c.raise_delta_height,
                end_enter=c.arm_elevated_enter,
                end_exit=c.arm_elevated_exit,
                increasing=True,
                code=0,
                reason="wrist height increases into elevated",
            )
            out += self._height_change(
                feats,
                name=f"LOWER_ARM_{side}",
                height_key=f"wrist_height_sh_{letter}",
                deriv_key=f"d_wrist_height_sh_{letter}",
                delta=c.raise_delta_height,
                end_enter=c.arm_down_enter,
                end_exit=c.arm_down_exit,
                increasing=False,
                code=0,
                reason="wrist height decreases into down",
            )

        # Separar / juntar manos
        out += self._hands_apart_together(feats, together=False)
        out += self._hands_apart_together(feats, together=True)

        # Abrir / cerrar brazos lateralmente
        out += self._open_close_arms(feats, opening=True)
        out += self._open_close_arms(feats, opening=False)

        return out

    # ------------------------------------------------------------------
    # FSMs genéricas
    # ------------------------------------------------------------------

    def _transition_increase(
        self,
        feats: FeatureSet,
        *,
        name: str,
        value_key: str,
        deriv_key: str,
        delta: float,
        end_enter: float,
        end_exit: float,
        start_below: float,
        code: int,
        reason: str,
    ) -> List[Segment]:
        """
        FSM: NOT_EXTENDED → EXTENDING → EXTENDED

        Requiere valor inicial < start_below, incremento >= delta,
        derivada positiva durante la transición, y fin en estado extended.
        """
        v = feats.get(value_key)
        dv = feats.get(deriv_key)
        n = len(v)
        mask = np.zeros(n, dtype=bool)
        state = "IDLE"
        start_i = 0
        start_val = np.nan
        min_frames = self.cfg.action_min_frames

        for i in range(n):
            vi, dvi = v[i], dv[i]
            if not np.isfinite(vi):
                state = "IDLE"
                continue

            if state == "IDLE":
                if vi < start_below:
                    state = "ARMED"
                    start_i = i
                    start_val = vi
            elif state == "ARMED":
                if np.isfinite(dvi) and dvi > 0 and vi > start_val + 5:
                    state = "EXTENDING"
                    start_i = i  # inicio del movimiento
                elif vi >= start_below:
                    state = "IDLE"
            elif state == "EXTENDING":
                mask[i] = True
                if vi >= end_enter and (vi - start_val) >= delta and (i - start_i) >= min_frames:
                    # completar segmento hasta aquí
                    state = "DONE"
                elif np.isfinite(dvi) and dvi < -abs(delta) * 0.1:
                    # abortar si vuelve atrás fuerte
                    mask[start_i : i + 1] = False
                    state = "IDLE"
            elif state == "DONE":
                # mantener unos frames del estado final breve, luego IDLE
                if vi < end_exit:
                    state = "IDLE"
                # no marcar DONE como acción continua

        segs = mask_to_segments(
            mask, name, code, reason=reason, feature=value_key,
            enter_threshold=end_enter, exit_threshold=end_exit,
        )
        # Filtrar: el segmento debe terminar cerca de extended
        kept: List[Segment] = []
        for s in segs:
            end_v = v[s.end_frame]
            start_v = v[s.start_frame]
            if (
                np.isfinite(end_v)
                and np.isfinite(start_v)
                and end_v >= end_exit
                and (end_v - start_v) >= delta * 0.8
            ):
                kept.append(s)
        return self.proc.process(kept, action=True)

    def _transition_decrease(
        self,
        feats: FeatureSet,
        *,
        name: str,
        value_key: str,
        deriv_key: str,
        delta: float,
        end_enter: float,
        end_exit: float,
        start_above: float,
        end_mode: str,
        code: int,
        reason: str,
    ) -> List[Segment]:
        v = feats.get(value_key)
        dv = feats.get(deriv_key)
        n = len(v)
        mask = np.zeros(n, dtype=bool)
        state = "IDLE"
        start_i = 0
        start_val = np.nan
        min_frames = self.cfg.action_min_frames

        for i in range(n):
            vi, dvi = v[i], dv[i]
            if not np.isfinite(vi):
                state = "IDLE"
                continue
            if state == "IDLE":
                if vi > start_above:
                    state = "ARMED"
                    start_i = i
                    start_val = vi
            elif state == "ARMED":
                if np.isfinite(dvi) and dvi < 0 and vi < start_val - 5:
                    state = "RETRACTING"
                    start_i = i
                elif vi <= start_above:
                    state = "IDLE"
            elif state == "RETRACTING":
                mask[i] = True
                if vi <= end_enter and (start_val - vi) >= delta and (i - start_i) >= min_frames:
                    state = "DONE"
                elif np.isfinite(dvi) and dvi > abs(delta) * 0.1:
                    mask[start_i : i + 1] = False
                    state = "IDLE"
            elif state == "DONE":
                if vi > end_exit:
                    state = "IDLE"

        segs = mask_to_segments(mask, name, code, reason=reason, feature=value_key)
        kept = []
        for s in segs:
            end_v, start_v = v[s.end_frame], v[s.start_frame]
            if (
                np.isfinite(end_v)
                and np.isfinite(start_v)
                and end_v <= end_exit
                and (start_v - end_v) >= delta * 0.8
            ):
                kept.append(s)
        return self.proc.process(kept, action=True)

    def _approach(
        self,
        feats: FeatureSet,
        *,
        name: str,
        dist_key: str,
        deriv_key: str,
        speed_key: str,
        near_enter: float,
        near_exit: float,
        code: int,
        reason: str,
    ) -> List[Segment]:
        """
        FSM: FAR → APPROACHING → NEAR

        Exige distancia inicial > near_exit, reducción >= approach_delta,
        derivada negativa, velocidad mínima, y final ≤ near_enter.
        """
        c = self.cfg
        d = feats.get(dist_key)
        dd = feats.get(deriv_key)
        sp = feats.get(speed_key)
        n = len(d)
        mask = np.zeros(n, dtype=bool)
        state = "IDLE"
        start_i = 0
        start_d = np.nan

        for i in range(n):
            di, ddi, spi = d[i], dd[i], sp[i]
            if not np.isfinite(di):
                state = "IDLE"
                continue
            if state == "IDLE":
                if di > near_exit:
                    state = "FAR"
                    start_d = di
                    start_i = i
            elif state == "FAR":
                if (
                    np.isfinite(ddi)
                    and ddi < 0
                    and np.isfinite(spi)
                    and spi >= c.min_action_speed_norm_s * 0.5
                    and di < start_d - 0.05
                ):
                    state = "APPROACHING"
                    start_i = i
                    start_d = di
                elif di <= near_exit:
                    # ya estaba cerca sin transición clara
                    state = "IDLE"
            elif state == "APPROACHING":
                mask[i] = True
                if di <= near_enter and (start_d - di) >= c.approach_delta_norm:
                    state = "NEAR"
                elif np.isfinite(ddi) and ddi > 0.05:
                    mask[start_i : i + 1] = False
                    state = "IDLE"
            elif state == "NEAR":
                if di > near_exit:
                    state = "IDLE"

        segs = mask_to_segments(
            mask, name, code, reason=reason, feature=dist_key,
            enter_threshold=near_enter, exit_threshold=near_exit,
        )
        kept = []
        for s in segs:
            if (
                np.isfinite(d[s.end_frame])
                and d[s.end_frame] <= near_exit
                and np.isfinite(d[s.start_frame])
                and (d[s.start_frame] - d[s.end_frame]) >= c.approach_delta_norm * 0.7
            ):
                kept.append(s)
        return self.proc.process(kept, action=True)

    def _height_change(
        self,
        feats: FeatureSet,
        *,
        name: str,
        height_key: str,
        deriv_key: str,
        delta: float,
        end_enter: float,
        end_exit: float,
        increasing: bool,
        code: int,
        reason: str,
    ) -> List[Segment]:
        h = feats.get(height_key)
        dh = feats.get(deriv_key)
        n = len(h)
        mask = np.zeros(n, dtype=bool)
        state = "IDLE"
        start_i = 0
        start_h = np.nan

        for i in range(n):
            hi, dhi = h[i], dh[i]
            if not np.isfinite(hi):
                state = "IDLE"
                continue
            if state == "IDLE":
                if increasing and hi < end_exit:
                    state = "ARMED"
                    start_h = hi
                    start_i = i
                elif not increasing and hi > end_exit:
                    state = "ARMED"
                    start_h = hi
                    start_i = i
            elif state == "ARMED":
                moving = (increasing and np.isfinite(dhi) and dhi > 0) or (
                    (not increasing) and np.isfinite(dhi) and dhi < 0
                )
                if moving:
                    state = "MOVING"
                    start_i = i
                    start_h = hi
                else:
                    # reset si deja de cumplir condición inicial
                    if increasing and hi >= end_exit:
                        state = "IDLE"
                    if not increasing and hi <= end_exit:
                        state = "IDLE"
            elif state == "MOVING":
                mask[i] = True
                done = False
                if increasing:
                    done = hi >= end_enter and (hi - start_h) >= delta
                else:
                    done = hi <= end_enter and (start_h - hi) >= delta
                if done:
                    state = "DONE"
                # abort
                if increasing and np.isfinite(dhi) and dhi < -0.05:
                    mask[start_i : i + 1] = False
                    state = "IDLE"
                if not increasing and np.isfinite(dhi) and dhi > 0.05:
                    mask[start_i : i + 1] = False
                    state = "IDLE"
            elif state == "DONE":
                state = "IDLE"

        segs = mask_to_segments(mask, name, code, reason=reason, feature=height_key)
        kept = []
        for s in segs:
            hs, he = h[s.start_frame], h[s.end_frame]
            if not (np.isfinite(hs) and np.isfinite(he)):
                continue
            if increasing and (he - hs) >= delta * 0.7 and he >= end_exit:
                kept.append(s)
            if (not increasing) and (hs - he) >= delta * 0.7 and he <= end_exit:
                kept.append(s)
        return self.proc.process(kept, action=True)

    def _hands_apart_together(self, feats: FeatureSet, *, together: bool) -> List[Segment]:
        c = self.cfg
        name = "JOIN_HANDS" if together else "SEPARATE_HANDS"
        d = feats.get("wrists_apart")
        dd = feats.get("d_wrists_apart")
        n = len(d)
        mask = np.zeros(n, dtype=bool)
        state = "IDLE"
        start_i = 0
        start_d = np.nan
        delta = c.hands_sep_delta_norm

        for i in range(n):
            di, ddi = d[i], dd[i]
            if not np.isfinite(di):
                state = "IDLE"
                continue
            if state == "IDLE":
                if together and di > c.wrists_together_exit_norm:
                    state = "ARMED"
                    start_d = di
                elif (not together) and di < c.wrists_apart_exit_norm:
                    state = "ARMED"
                    start_d = di
            elif state == "ARMED":
                if together and np.isfinite(ddi) and ddi < 0:
                    state = "MOVING"
                    start_i = i
                    start_d = di
                elif (not together) and np.isfinite(ddi) and ddi > 0:
                    state = "MOVING"
                    start_i = i
                    start_d = di
            elif state == "MOVING":
                mask[i] = True
                if together and di <= c.wrists_together_enter_norm and (start_d - di) >= delta:
                    state = "DONE"
                elif (not together) and di >= c.wrists_apart_enter_norm and (di - start_d) >= delta:
                    state = "DONE"
            elif state == "DONE":
                state = "IDLE"

        segs = mask_to_segments(mask, name, 0, reason="wrists distance transition", feature="wrists_apart")
        return self.proc.process(segs, action=True)

    def _open_close_arms(self, feats: FeatureSet, *, opening: bool) -> List[Segment]:
        """
        OPEN: parte de brazos relativamente cerrados y aumenta open_both.
        CLOSE: parte de brazos relativamente abiertos y disminuye open_both.
        No marca frames solo por estar en el estado final.
        """
        c = self.cfg
        name = "OPEN_ARMS_LATERAL" if opening else "CLOSE_ARMS_LATERAL"
        v = feats.get("open_both")
        dv = feats.get("d_open_both")
        n = len(v)
        mask = np.zeros(n, dtype=bool)
        state = "IDLE"
        start_i = 0
        start_v = np.nan
        delta = c.open_close_delta

        for i in range(n):
            vi, dvi = v[i], dv[i]
            if not np.isfinite(vi):
                state = "IDLE"
                continue
            if state == "IDLE":
                # Condición inicial: no estar ya en el destino
                if opening and vi < c.arms_open_exit:
                    state = "ARMED"
                    start_v = vi
                elif (not opening) and vi > c.arms_open_enter:
                    state = "ARMED"
                    start_v = vi
            elif state == "ARMED":
                if opening and np.isfinite(dvi) and dvi > 0 and vi > start_v + 0.02:
                    state = "MOVING"
                    start_i = i
                    start_v = vi
                elif (not opening) and np.isfinite(dvi) and dvi < 0 and vi < start_v - 0.02:
                    state = "MOVING"
                    start_i = i
                    start_v = vi
                elif opening and vi >= c.arms_open_exit:
                    state = "IDLE"
                elif (not opening) and vi <= c.arms_open_enter:
                    state = "IDLE"
            elif state == "MOVING":
                mask[i] = True
                if opening and (vi - start_v) >= delta and vi >= c.arms_open_enter * 0.8:
                    state = "DONE"
                elif (not opening) and (start_v - vi) >= delta and vi <= c.arms_open_exit:
                    state = "DONE"
                # abortar si se invierte el sentido
                elif opening and np.isfinite(dvi) and dvi < -0.05:
                    mask[start_i : i + 1] = False
                    state = "IDLE"
                elif (not opening) and np.isfinite(dvi) and dvi > 0.05:
                    mask[start_i : i + 1] = False
                    state = "IDLE"
            elif state == "DONE":
                state = "IDLE"

        segs = mask_to_segments(mask, name, 0, reason="lateral arm opening change", feature="open_both")
        kept = []
        for s in segs:
            vs, ve = v[s.start_frame], v[s.end_frame]
            if not (np.isfinite(vs) and np.isfinite(ve)):
                continue
            if opening and (ve - vs) >= delta * 0.7:
                kept.append(s)
            if (not opening) and (vs - ve) >= delta * 0.7:
                kept.append(s)
        return self.proc.process(kept, action=True)


class ActionAnalysisPipeline:
    """Orquesta features → estados → acciones → segmentos finales."""

    def __init__(self, config: DetectionConfig | None = None):
        self.cfg = config or DetectionConfig()

    def run(self, poses: np.ndarray, fps: float) -> Tuple[List[Segment], FeatureSet]:
        from .features import FeatureExtractor, PoseSequence

        seq = PoseSequence(poses=poses, fps=fps)
        feats = FeatureExtractor(self.cfg).extract(seq)
        states = StateDetector(self.cfg, fps).detect_all(feats)
        actions = ActionDetector(self.cfg, fps).detect_all(feats)
        segments = sorted(states + actions, key=lambda s: (s.start_frame, s.name))
        return segments, feats
