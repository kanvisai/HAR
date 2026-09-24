"""Procesamiento temporal de máscaras booleanas → segmentos."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .config import DetectionConfig, Segment


def hysteresis_mask(
    values: np.ndarray,
    enter: float,
    exit: float,
    *,
    mode: str = "above",
) -> np.ndarray:
    """
    Aplica histeresis frame a frame.

    mode="above": entra si value > enter, sale si value < exit (enter >= exit)
    mode="below": entra si value < enter, sale si value > exit (enter <= exit)
    mode="abs_below": entra si |value| < enter, sale si |value| > exit
    """
    n = len(values)
    out = np.zeros(n, dtype=bool)
    state = False
    for i, v in enumerate(values):
        if not np.isfinite(v):
            # Mantener estado o apagar? Conservador: apagar
            state = False
            out[i] = False
            continue
        if mode == "above":
            if not state and v > enter:
                state = True
            elif state and v < exit:
                state = False
        elif mode == "below":
            if not state and v < enter:
                state = True
            elif state and v > exit:
                state = False
        elif mode == "abs_below":
            av = abs(v)
            if not state and av < enter:
                state = True
            elif state and av > exit:
                state = False
        else:
            raise ValueError(f"mode desconocido: {mode}")
        out[i] = state
    return out


def mask_to_segments(
    mask: np.ndarray,
    name: str,
    code: int,
    *,
    reason: str = "",
    feature: str = "",
    enter_threshold: float | None = None,
    exit_threshold: float | None = None,
) -> List[Segment]:
    """Convierte máscara booleana en segmentos [start, end] inclusivos."""
    mask = np.asarray(mask, dtype=bool)
    segs: List[Segment] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        segs.append(
            Segment(
                name=name,
                start_frame=i,
                end_frame=j,
                code=code,
                reason=reason,
                feature=feature,
                enter_threshold=enter_threshold,
                exit_threshold=exit_threshold,
            )
        )
        i = j + 1
    return segs


class SegmentProcessor:
    """Filtra por duración mínima y fusiona gaps cortos."""

    def __init__(self, config: DetectionConfig, fps: float):
        self.cfg = config
        self.fps = fps

    def merge_gaps(self, segments: Sequence[Segment], gap_s: float | None = None) -> List[Segment]:
        if not segments:
            return []
        gap_s = self.cfg.merge_gap_s if gap_s is None else gap_s
        gap_frames = max(0, int(round(gap_s * self.fps)))
        # Agrupar por nombre+código
        by_key: dict[tuple, List[Segment]] = {}
        for s in segments:
            by_key.setdefault((s.name, s.code), []).append(s)

        merged: List[Segment] = []
        for (_, _), group in by_key.items():
            group = sorted(group, key=lambda s: s.start_frame)
            cur = group[0]
            for nxt in group[1:]:
                if nxt.start_frame - cur.end_frame - 1 <= gap_frames:
                    cur = Segment(
                        name=cur.name,
                        start_frame=cur.start_frame,
                        end_frame=max(cur.end_frame, nxt.end_frame),
                        code=cur.code,
                        reason=cur.reason or nxt.reason,
                        feature=cur.feature or nxt.feature,
                        enter_threshold=cur.enter_threshold,
                        exit_threshold=cur.exit_threshold,
                    )
                else:
                    merged.append(cur)
                    cur = nxt
            merged.append(cur)
        return sorted(merged, key=lambda s: (s.start_frame, s.name))

    def filter_min_duration(
        self, segments: Sequence[Segment], min_s: float, *, action: bool = False
    ) -> List[Segment]:
        if min_s is None:
            min_s = (
                self.cfg.min_action_duration_s
                if action
                else self.cfg.min_state_duration_s
            )
        return [s for s in segments if s.duration_s(self.fps) >= min_s]

    def process(
        self, segments: Sequence[Segment], *, action: bool = False
    ) -> List[Segment]:
        min_s = (
            self.cfg.min_action_duration_s if action else self.cfg.min_state_duration_s
        )
        segs = self.merge_gaps(segments)
        segs = self.filter_min_duration(segs, min_s, action=action)
        # Re-merge tras filtrado no es necesario; merge primero evita gaps
        # Reaplicar merge tras filter por si quedan adyacentes (no deberían)
        segs = self.merge_gaps(segs)
        return segs
