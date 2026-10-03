"""
Gestures from the 8-point skeleton.

extend_arm, retract_arm, turn_torso, crouch, steady_wrist_waist.

A point that is missing in the npy is not measured and is not filled in.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from har_actions.pose import ARM_SIDE, HIP_L, HIP_R, SHOULDER_L, SHOULDER_R, format_time

ACTION_CONFIG = Path(__file__).resolve().parents[1] / "actions_config.json"

NOT_OBSERVABLE: tuple[str, ...] = ()


@dataclass
class Action:
    id: str
    name: str
    side: str
    frame_start: int
    frame_end: int
    detail: str

    def to_dict(self, fps: float) -> dict:
        data = asdict(self)
        data["t_start"] = format_time(self.frame_start / fps)
        data["t_end"] = format_time((self.frame_end + 1) / fps)
        data["duration_s"] = round((self.frame_end - self.frame_start + 1) / fps, 2)
        return data


def detect(poses: np.ndarray, fps: float, aspect: float = 16 / 9) -> list[Action]:
    """Smoothed poses, shape (T, 8, 2). aspect is the frame width divided by its height."""
    features = _features(poses, fps, aspect)
    actions: list[Action] = []
    actions += _extend_arm(features)
    actions += _extend_forearm(features)
    actions += _retract_arm(features)
    actions += _turn_torso(features)
    actions += _crouch(features)
    actions += _steady_wrist_waist(features)
    steady = [action for action in actions if action.id == "steady_wrist_waist"]
    rest = [action for action in actions if action.id != "steady_wrist_waist"]
    actions = _merge(rest, gap=4) + _merge(steady, gap=2)
    config = load_action_config()
    actions = [action for action in actions if action_enabled(action.id, config)]
    actions.sort(key=lambda item: (item.frame_start, item.frame_end, item.id, item.side))
    return actions


def load_action_config(path: Path | None = None) -> dict[str, bool]:
    """id -> enabled. If the file is missing, every action stays enabled."""
    config_path = path or ACTION_CONFIG
    if not config_path.is_file():
        return {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    enabled: dict[str, bool] = {}
    for item in data.get("actions", []):
        if "id" in item:
            enabled[item["id"]] = bool(item.get("enabled", True))
    return enabled


def action_enabled(action_id: str, config: dict[str, bool] | None = None) -> bool:
    mapping = load_action_config() if config is None else config
    return mapping.get(action_id, True)


def disabled_actions() -> list[str]:
    return [action_id for action_id, enabled in load_action_config().items() if not enabled]


def actions_at_frame(actions: list[Action], frame: int) -> list[Action]:
    return [action for action in actions if action.frame_start <= frame <= action.frame_end]


def report(actions: list[Action], fps: float) -> str:
    if not actions:
        return "No gesture was marked in this clip."
    lines = [f"{len(actions)} gesture(s):", ""]
    for action in actions:
        data = action.to_dict(fps)
        lines.append(f"  {data['t_start']} → {data['t_end']}   {action.name}   [{action.side}]")
        lines.append(f"      {action.detail}")
        lines.append(f"      frames {action.frame_start}–{action.frame_end}  ({data['duration_s']} s)")
        lines.append("")
    return "\n".join(lines).rstrip()


def _features(poses: np.ndarray, fps: float, aspect: float = 16 / 9) -> dict:
    p = np.array(poses, dtype=np.float64, copy=True)
    p[:, :, 0] *= aspect
    t = len(p)
    both_shoulders = np.isfinite(p[:, SHOULDER_L]).all(axis=1) & np.isfinite(p[:, SHOULDER_R]).all(axis=1)
    both_hips = np.isfinite(p[:, HIP_L]).all(axis=1) & np.isfinite(p[:, HIP_R]).all(axis=1)
    shoulder_c = 0.5 * (p[:, SHOULDER_L] + p[:, SHOULDER_R])
    hip_c = 0.5 * (p[:, HIP_L] + p[:, HIP_R])
    torso = np.linalg.norm(shoulder_c - hip_c, axis=1)
    width = np.linalg.norm(p[:, SHOULDER_L] - p[:, SHOULDER_R], axis=1)
    torso_ok = both_shoulders & both_hips & (torso > 0.04) & (torso < 0.85)
    torso_ok &= hip_c[:, 1] > shoulder_c[:, 1] - 0.02
    scale = np.maximum(torso, 1e-3)
    frontal = np.divide(width, scale, out=np.full(t, np.nan), where=torso_ok)
    torso_vertical = hip_c[:, 1] - shoulder_c[:, 1]

    arms = {}
    for side, (shoulder, elbow, wrist, hip) in ARM_SIDE.items():
        w = p[:, wrist]
        s = p[:, shoulder]
        e = p[:, elbow]
        hip_pt = p[:, hip]
        finite = (
            np.isfinite(w).all(axis=1)
            & np.isfinite(s).all(axis=1)
            & np.isfinite(e).all(axis=1)
            & np.isfinite(hip_pt).all(axis=1)
        )
        upper = np.linalg.norm(e - s, axis=1)
        forearm = np.linalg.norm(w - e, axis=1)
        angle = _angle(s, e, w)
        ok = torso_ok & finite & (upper > 0.10 * scale) & (forearm > 0.08 * scale) & (angle > 40)
        torso_height = np.maximum(hip_pt[:, 1] - s[:, 1], 1e-3)
        height = (hip_pt[:, 1] - w[:, 1]) / torso_height
        arms[side] = {
            "ok": ok,
            "angle": angle,
            "forearm": forearm / scale,
            "dist_hip": np.linalg.norm(w - hip_pt, axis=1) / scale,
            "dist_axis": _segment_distance(w, s, hip_pt) / scale,
            "dist_shoulder": np.linalg.norm(w - s, axis=1) / scale,
            "height": height,
        }

    return {
        "fps": fps,
        "n": t,
        "scale": scale,
        "frontal": frontal,
        "torso_vertical": torso_vertical,
        "torso_ok": torso_ok,
        "arms": arms,
    }


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = np.sum(segment * segment, axis=1) + 1e-8
    with np.errstate(invalid="ignore"):
        t = np.clip(np.sum((point - start) * segment, axis=1) / denom, 0.0, 1.0)
    projection = start + t[:, None] * segment
    return np.linalg.norm(point - projection, axis=1)


def _spans(mask: np.ndarray, minimum: int, gap: int = 2) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool).copy()
    if gap > 0 and m.any():
        i = 0
        n = len(m)
        while i < n:
            if m[i]:
                i += 1
                continue
            j = i
            while j < n and not m[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= gap:
                m[i:j] = True
            i = j
    spans = []
    i = 0
    n = len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        if (j - i) >= minimum:
            spans.append((i, j - 1))
        i = j
    return spans


def _merge(actions: list[Action], gap: int) -> list[Action]:
    groups: dict[tuple[str, str], list[Action]] = {}
    for action in actions:
        groups.setdefault((action.id, action.side), []).append(action)
    output: list[Action] = []
    for group in groups.values():
        group.sort(key=lambda item: item.frame_start)
        current = group[0]
        for nxt in group[1:]:
            if nxt.frame_start <= current.frame_end + gap:
                current = Action(
                    current.id,
                    current.name,
                    current.side,
                    current.frame_start,
                    max(current.frame_end, nxt.frame_end),
                    current.detail,
                )
            else:
                output.append(current)
                current = nxt
        output.append(current)
    return output


def _windows(n: int, width: int, step: int = 1):
    limit = n - width
    i = 0
    while i <= limit:
        yield i, i + width - 1
        i += step


def _collapse_at_the_end(arm: dict, i: int, j: int, fps: float, closing: bool) -> bool:
    angle = arm["angle"]
    tail = max(2, int(round(0.20 * fps)))
    earlier = max(i, j - tail)
    if not np.isfinite(angle[i]) or not np.isfinite(angle[j]) or not np.isfinite(angle[earlier]):
        return False
    if closing:
        late = float(angle[earlier] - angle[j])
        total = float(angle[i] - angle[j])
    else:
        late = float(angle[j] - angle[earlier])
        total = float(angle[j] - angle[i])
    if total <= 0 or late / total <= 0.55:
        return False
    hold = max(3, int(round(0.24 * fps)))
    after = arm["ok"][j + 1 : j + 1 + hold]
    return len(after) == 0 or float(after.mean()) < 0.5


def _gesture_widths(fps: float):
    short = max(5, int(round(0.40 * fps)))
    long = max(short + 1, int(round(1.15 * fps)))
    return range(short, long + 1, 3)


def _extend_arm(features: dict) -> list[Action]:
    actions = []
    for side, arm in features["arms"].items():
        dist = arm["dist_shoulder"]
        angle = arm["angle"]
        for width in _gesture_widths(features["fps"]):
            for i, j in _windows(features["n"], width, step=2):
                if not arm["ok"][i] or not arm["ok"][j]:
                    continue
                if arm["ok"][i : j + 1].mean() < 0.75:
                    continue
                if angle[j] - angle[i] < 35:
                    continue
                if angle[i] > 125:
                    continue
                # From above, the hand can finish far from the body while the
                # elbow stays under 130 degrees. That is still the arm extending.
                reaches_out = (
                    arm["dist_axis"][j] >= 0.55
                    and arm["dist_axis"][j] - arm["dist_axis"][i] >= 0.30
                    and angle[j] >= 100
                )
                if angle[j] < 130 and not reaches_out:
                    continue
                if dist[j] - dist[i] < 0.10:
                    continue
                mid = (i + j) // 2
                if angle[mid] - angle[i] < 12:
                    continue
                if angle[mid] < angle[i] - 15 or angle[j] < angle[mid] - 8:
                    continue
                if _collapse_at_the_end(arm, i, j, features["fps"], closing=False):
                    continue
                actions.append(
                    Action(
                        "extend_arm",
                        "Extend the arm",
                        side,
                        i,
                        j,
                        (
                            f"The {side} elbow opens and the hand moves away from the shoulder "
                            f"(the angle goes from {angle[i]:.0f}° to {angle[j]:.0f}°)."
                        ),
                    )
                )
        actions.extend(_reach_out(features, side, arm))
    return actions


def _reach_out(features: dict, side: str, arm: dict) -> list[Action]:
    """A straight arm that leaves the torso. From above, the elbow is already open."""
    actions = []
    away = arm["dist_axis"]
    angle = arm["angle"]
    for width in _gesture_widths(features["fps"]):
        for i, j in _windows(features["n"], width, step=2):
            if not arm["ok"][i] or not arm["ok"][j]:
                continue
            if arm["ok"][i : j + 1].mean() < 0.75:
                continue
            if away[i] > 0.35 or away[j] < 0.50:
                continue
            if away[j] - away[i] < 0.35:
                continue
            if angle[i] < 130 or angle[j] < 125:
                continue
            mid = (i + j) // 2
            if away[mid] - away[i] < 0.12:
                continue
            actions.append(
                Action(
                    "extend_arm",
                    "Extend the arm",
                    side,
                    i,
                    j,
                    f"The {side} arm reaches out, away from the body.",
                )
            )
    return actions


def _extend_forearm(features: dict) -> list[Action]:
    """The elbow opens and the forearm swings out, without the whole arm ending straight."""
    actions = []
    for side, arm in features["arms"].items():
        angle = arm["angle"]
        away = arm["dist_axis"]
        dist = arm["dist_shoulder"]
        for width in _gesture_widths(features["fps"]):
            for i, j in _windows(features["n"], width, step=2):
                if not arm["ok"][i] or not arm["ok"][j]:
                    continue
                if arm["ok"][i : j + 1].mean() < 0.70:
                    continue
                opening = float(angle[j] - angle[i])
                if opening < 30:
                    continue
                if angle[i] > 100 or angle[j] < 95 or angle[j] > 135:
                    continue
                # A fully straightened arm, or a hand that ends far from the body, is extend_arm.
                if angle[j] >= 130 and opening >= 35 and dist[j] - dist[i] >= 0.10:
                    continue
                if away[j] >= 0.55 and away[j] - away[i] >= 0.30:
                    continue
                if away[j] - away[i] < 0.15 and dist[j] - dist[i] < 0.08:
                    continue
                mid = (i + j) // 2
                if angle[mid] - angle[i] < 10:
                    continue
                if _collapse_at_the_end(arm, i, j, features["fps"], closing=False):
                    continue
                actions.append(
                    Action(
                        "extend_forearm",
                        "Extend the forearm",
                        side,
                        i,
                        j,
                        (
                            f"The {side} forearm swings out as the elbow opens "
                            f"(the angle goes from {angle[i]:.0f}° to {angle[j]:.0f}°)."
                        ),
                    )
                )
    return actions


def _retract_arm(features: dict) -> list[Action]:
    actions = []
    for side, arm in features["arms"].items():
        dist = arm["dist_shoulder"]
        angle = arm["angle"]
        for width in _gesture_widths(features["fps"]):
            for i, j in _windows(features["n"], width, step=2):
                if not arm["ok"][i] or not arm["ok"][j]:
                    continue
                if arm["ok"][i : j + 1].mean() < 0.75:
                    continue
                if angle[i] - angle[j] < 35:
                    continue
                if angle[i] < 145 or angle[j] > 120 or angle[j] <= 45:
                    continue
                if dist[i] - dist[j] < 0.10:
                    continue
                if arm["dist_axis"][j] - arm["dist_axis"][i] > 0.20:
                    continue
                mid = (i + j) // 2
                if angle[i] - angle[mid] < 12:
                    continue
                if angle[mid] > angle[i] + 15 or angle[j] > angle[mid] + 8:
                    continue
                if _collapse_at_the_end(arm, i, j, features["fps"], closing=True):
                    continue
                actions.append(
                    Action(
                        "retract_arm",
                        "Retract the arm",
                        side,
                        i,
                        j,
                        (
                            f"The {side} elbow closes and the hand comes back toward the shoulder "
                            f"(the angle goes from {angle[i]:.0f}° to {angle[j]:.0f}°)."
                        ),
                    )
                )
        actions.extend(_come_back(features, side, arm))
    return actions


def _come_back(features: dict, side: str, arm: dict) -> list[Action]:
    actions = []
    away = arm["dist_axis"]
    ok = arm["ok"]
    for width in _gesture_widths(features["fps"]):
        for i, j in _windows(features["n"], width, step=2):
            if not ok[i] or not ok[j]:
                continue
            if float(ok[i : j + 1].mean()) < 0.45:
                continue
            if away[i] < 0.45 or away[j] > 0.25:
                continue
            if away[i] - away[j] < 0.30:
                continue
            actions.append(
                Action(
                    "retract_arm",
                    "Retract the arm",
                    side,
                    i,
                    j,
                    f"The {side} arm comes back in toward the body.",
                )
            )
    return actions


def _turn_torso(features: dict) -> list[Action]:
    """The shoulders go from facing the camera to a side view, or the other way."""
    ratio = features["frontal"]
    ok = features["torso_ok"]
    actions = []
    fps = features["fps"]
    short = max(5, int(round(0.45 * fps)))
    long = max(short + 1, int(round(1.20 * fps)))
    for width in range(short, long + 1, 3):
        for i, j in _windows(features["n"], width, step=2):
            if ok[i : j + 1].mean() < 0.8:
                continue
            if not np.isfinite(ratio[i]) or not np.isfinite(ratio[j]):
                continue
            change = float(ratio[j] - ratio[i])
            turning_away = ratio[i] >= 0.48 and ratio[j] <= 0.30 and change <= -0.20
            turning_back = ratio[i] <= 0.30 and ratio[j] >= 0.48 and change >= 0.20
            if not turning_away and not turning_back:
                continue
            mid = (i + j) // 2
            if not np.isfinite(ratio[mid]):
                continue
            if turning_away and ratio[i] - ratio[mid] < 0.08:
                continue
            if turning_back and ratio[mid] - ratio[i] < 0.08:
                continue
            direction = "sideways" if turning_away else "back toward the camera"
            actions.append(
                Action(
                    "turn_torso",
                    "Turn the torso",
                    "both",
                    i,
                    j,
                    (
                        f"The shoulders turn {direction} "
                        f"(the shoulder width goes from {ratio[i]:.2f} to {ratio[j]:.2f} of the torso)."
                    ),
                )
            )
    return actions


def _crouch(features: dict) -> list[Action]:
    """A small bend: the torso gets shorter in the image while the person stays about the same size."""
    vertical = features["torso_vertical"]
    scale = features["scale"]
    ok = features["torso_ok"]
    actions = []
    fps = features["fps"]
    short = max(5, int(round(0.40 * fps)))
    long = max(short + 1, int(round(0.90 * fps)))
    for width in range(short, long + 1, 2):
        for i, j in _windows(features["n"], width, step=2):
            if ok[i : j + 1].mean() < 0.75:
                continue
            if scale[i] <= 0 or not np.isfinite(vertical[i]) or not np.isfinite(vertical[j]):
                continue
            if not (0.88 <= scale[j] / scale[i] <= 1.12):
                continue
            drop = float(vertical[i] - vertical[j])
            if drop < 0.012 or drop < 0.18 * float(vertical[i]):
                continue
            mid = (i + j) // 2
            if not np.isfinite(vertical[mid]) or vertical[i] - vertical[mid] < 0.4 * drop:
                continue
            actions.append(
                Action(
                    "crouch",
                    "Crouch a little",
                    "both",
                    i,
                    j,
                    "The torso shortens in the image. The person bends down a little without walking off.",
                )
            )
    return actions


def _steady_wrist_waist(features: dict) -> list[Action]:
    """The wrist stays at a nearly fixed distance from the hip."""
    fps = features["fps"]
    window = max(8, int(round(1.0 * fps)))
    minimum = window
    actions = []
    for side, arm in features["arms"].items():
        distance = arm["dist_hip"]
        ok = arm["ok"]
        mask = np.zeros(features["n"], dtype=bool)
        limit = features["n"] - window + 1
        for i in range(max(0, limit)):
            stretch = distance[i : i + window]
            if float(ok[i : i + window].mean()) < 0.85:
                continue
            if not np.isfinite(stretch).all():
                continue
            if float(stretch.max() - stretch.min()) > 0.06:
                continue
            mask[i : i + window] = True
        for i, j in _spans(mask, minimum, gap=0):
            held = distance[i : j + 1]
            actions.append(
                Action(
                    "steady_wrist_waist",
                    "Steady wrist-to-waist distance",
                    side,
                    i,
                    j,
                    (
                        f"The {side} wrist stays at about the same distance from the hip "
                        f"({np.median(held):.2f} torso lengths)."
                    ),
                )
            )
    return actions
