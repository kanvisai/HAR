"""
Gestures for aisle categories 0, 1 and 5.

walk, stop, turn_torso, arm_swing, carry, hand_up.

A point that is missing in the npy is not measured and is not filled in.
There is no head point, so a downward look is not measured.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from har_actions.pose import ARM_SIDE, HIP_L, HIP_R, SHOULDER_L, SHOULDER_R, format_time

ACTION_CONFIG = Path(__file__).resolve().parents[1] / "actions_config.json"

NOT_OBSERVABLE: tuple[str, ...] = ("looking down",)


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
    walk_mask = _motion_mask(features, walking=True)
    stop_mask = _motion_mask(features, walking=False)
    stop_mask &= ~_dilate(walk_mask, 3)
    actions: list[Action] = []
    actions += _from_mask(features, walk_mask, "walk", "Walk", "The hips move along the aisle.")
    actions += _from_mask(features, stop_mask, "stop", "Stop", "The hips stay in place.")
    actions += _turn_torso(features)
    actions += _arm_swing(features, walk_mask)
    actions += _carry(features, walk_mask)
    actions += _hand_up(features)
    actions += _hand_to_body(features)
    actions = _merge(actions, gap=4)
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


GESTURE_IDS = ("walk", "stop", "turn_torso", "arm_swing", "carry", "hand_up", "hand_to_body")


def gesture_seconds(actions: list[Action], fps: float) -> dict[str, float]:
    """Sum each gesture. Overlapping segments of the same id both count."""
    totals = {name: 0.0 for name in GESTURE_IDS}
    for action in actions:
        totals[action.id] = totals.get(action.id, 0.0) + (action.frame_end - action.frame_start + 1) / fps
    return totals


def visible_seconds(poses: np.ndarray, fps: float) -> float:
    """Seconds in which at least one keypoint is present."""
    return int(np.isfinite(poses).any(axis=(1, 2)).sum()) / fps


def evaluate_category(poses: np.ndarray, actions: list[Action], fps: float) -> dict:
    """Choose c0, c1, c5, c100 or inconclusive from the gesture durations.

    The first matching rule wins. turn_torso, arm_swing and hand_up are measured
    and do not choose the category.
    """
    visible = visible_seconds(poses, fps)
    raw = gesture_seconds(actions, fps)
    shown = {key: round(value, 2) for key, value in raw.items()}
    steps: list[dict] = []
    category = None
    decided_by = None
    reason = None

    def consider(order: int, rule: str, uses: dict, test: bool, label: str, because: str) -> None:
        nonlocal category, decided_by, reason
        reached = category is None
        matched = bool(test) if reached else False
        steps.append({
            "order": order,
            "rule": rule,
            "uses": uses,
            "reached": reached,
            "matched": matched,
            "category_if_matched": label,
        })
        if reached and matched:
            category = label
            decided_by = rule
            reason = because

    consider(
        1,
        "hand_to_body > 0 s",
        {"hand_to_body_s": shown["hand_to_body"]},
        raw["hand_to_body"] > 0,
        "inconclusive",
        "A hand is out from the torso and also against the hip. That pattern is not called c0, c1 or c5.",
    )
    consider(
        2,
        "visible < 2.0 s",
        {"visible_s": round(visible, 2)},
        visible < 2.0,
        "inconclusive",
        "The skeleton is visible for less than 2.0 seconds.",
    )
    consider(
        3,
        "stop >= 2.5 s and walk < 2.0 s",
        {"stop_s": shown["stop"], "walk_s": shown["walk"]},
        raw["stop"] >= 2.5 and raw["walk"] < 2.0,
        "c100",
        "The person stays still: the stop lasts at least 2.5 seconds and the walk lasts under 2.0 seconds.",
    )
    consider(
        4,
        "stop >= 2.5 s and walk >= 2.0 s",
        {"stop_s": shown["stop"], "walk_s": shown["walk"]},
        raw["stop"] >= 2.5 and raw["walk"] >= 2.0,
        "c1",
        "The person walked and stopped. The stop lasts at least 2.5 seconds and the walk lasts at least 2.0 seconds.",
    )
    consider(
        5,
        "walk >= 2.0 s and carry >= 2.0 s",
        {"walk_s": shown["walk"], "carry_s": shown["carry"]},
        raw["walk"] >= 2.0 and raw["carry"] >= 2.0,
        "c5",
        "The person keeps walking with one hand steady, as if carrying something.",
    )
    consider(
        6,
        "walk >= 2.0 s",
        {"walk_s": shown["walk"]},
        raw["walk"] >= 2.0,
        "c0",
        "The person keeps walking. The carry does not last 2.0 seconds, so the arms count as free.",
    )
    consider(
        7,
        "anything shorter",
        {"walk_s": shown["walk"], "stop_s": shown["stop"], "carry_s": shown["carry"]},
        True,
        "inconclusive",
        "The walk, the stop and the carry are all shorter than the thresholds.",
    )

    return {
        "category": category,
        "decided_by": decided_by,
        "reason": reason,
        "visible_s": round(visible, 2),
        "durations_s": shown,
        "evaluation": steps,
        "not_used_for_the_category": ["turn_torso", "arm_swing", "hand_up", "looking down"],
        "note": "The first matching rule wins. c0 and c5 may swap. Looking down is not measured: the skeleton has no head point.",
    }


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
            "dist_hip": np.linalg.norm(w - hip_pt, axis=1) / scale,
            "dist_axis": _segment_distance(w, s, hip_pt) / scale,
            "height": height,
        }

    return {
        "fps": fps,
        "n": t,
        "scale": scale,
        "frontal": frontal,
        "torso_ok": torso_ok,
        "hip": hip_c,
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


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.copy()
    out = mask.copy()
    idx = np.flatnonzero(mask)
    for shift in range(1, radius + 1):
        out[np.clip(idx - shift, 0, len(mask) - 1)] = True
        out[np.clip(idx + shift, 0, len(mask) - 1)] = True
    return out


def _motion_mask(features: dict, walking: bool) -> np.ndarray:
    """True where the hips are travelling, or where they stay put."""
    hip = features["hip"]
    scale = features["scale"]
    ok = features["torso_ok"]
    n = features["n"]
    window = max(8, int(round((0.90 if walking else 1.10) * features["fps"])))
    mask = np.zeros(n, dtype=bool)
    limit = n - window + 1
    for i in range(max(0, limit)):
        if ok[i : i + window].mean() < 0.70:
            continue
        pts = hip[i : i + window]
        finite = np.isfinite(pts).all(axis=1)
        if finite.mean() < 0.70:
            continue
        moved = float(np.linalg.norm(pts[-1] - pts[0]) / max(float(np.median(scale[i : i + window])), 1e-3))
        center = np.median(pts[finite], axis=0)
        radius = float(np.max(np.linalg.norm(pts[finite] - center, axis=1)) / max(float(np.median(scale[i : i + window])), 1e-3))
        if walking and moved >= 0.45 and radius >= 0.28:
            mask[i : i + window] = True
        if not walking and moved <= 0.16 and radius <= 0.15:
            mask[i : i + window] = True
    return mask


def _from_mask(features: dict, mask: np.ndarray, action_id: str, name: str, detail: str) -> list[Action]:
    minimum = max(8, int(round((0.90 if action_id == "walk" else 1.10) * features["fps"])))
    actions = []
    for i, j in _spans(mask, minimum, gap=3):
        actions.append(Action(action_id, name, "both", i, j, detail))
    return actions


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
            direction = "sideways, toward the shelf" if turning_away else "back toward the aisle"
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


def _arm_swing(features: dict, walk_mask: np.ndarray) -> list[Action]:
    """A free arm swings while the person walks. The wrist distance to the hip goes up and down."""
    fps = features["fps"]
    window = max(8, int(round(0.80 * fps)))
    actions = []
    for side, arm in features["arms"].items():
        distance = arm["dist_hip"]
        ok = arm["ok"]
        mask = np.zeros(features["n"], dtype=bool)
        limit = features["n"] - window + 1
        for i in range(max(0, limit)):
            if walk_mask[i : i + window].mean() < 0.60:
                continue
            if ok[i : i + window].mean() < 0.70:
                continue
            stretch = distance[i : i + window]
            if not np.isfinite(stretch).all():
                continue
            if float(stretch.max() - stretch.min()) < 0.16:
                continue
            delta = np.diff(stretch)
            signs = np.sign(delta)
            signs = signs[signs != 0]
            if len(signs) < 2 or int(np.sum(signs[1:] * signs[:-1] < 0)) < 1:
                continue
            mask[i : i + window] = True
        for i, j in _spans(mask, window, gap=2):
            actions.append(
                Action(
                    "arm_swing",
                    "Swing the arm",
                    side,
                    i,
                    j,
                    f"The {side} arm swings with the step. The hand is not holding anything still.",
                )
            )
    return actions


def _carry(features: dict, walk_mask: np.ndarray) -> list[Action]:
    """A bent arm stays put relative to the body while the person walks."""
    fps = features["fps"]
    window = max(8, int(round(1.00 * fps)))
    actions = []
    for side, arm in features["arms"].items():
        distance = arm["dist_hip"]
        away = arm["dist_axis"]
        height = arm["height"]
        ok = arm["ok"]
        mask = np.zeros(features["n"], dtype=bool)
        limit = features["n"] - window + 1
        for i in range(max(0, limit)):
            if walk_mask[i : i + window].mean() < 0.60:
                continue
            if ok[i : i + window].mean() < 0.80:
                continue
            stretch = distance[i : i + window]
            held_out = away[i : i + window]
            lifted = height[i : i + window]
            if not np.isfinite(stretch).all() or not np.isfinite(held_out).all():
                continue
            if float(stretch.max() - stretch.min()) > 0.07:
                continue
            if float(np.median(held_out)) < 0.34 and float(np.nanmedian(lifted)) < 0.42:
                continue
            mask[i : i + window] = True
        for i, j in _spans(mask, window, gap=2):
            actions.append(
                Action(
                    "carry",
                    "Carry",
                    side,
                    i,
                    j,
                    f"The {side} hand stays with the body while the person walks, as if holding a basket, a bag or a product.",
                )
            )
    return actions


def _hand_to_body(features: dict) -> list[Action]:
    """The hand leaves the torso and also spends time against the hip.

    Category 6 does both. A walk-through, a stop to look, or a carried bag
    usually does only one of them.
    """
    fps = features["fps"]
    n = features["n"]
    far = np.zeros(n, dtype=bool)
    close = np.zeros(n, dtype=bool)
    for arm in features["arms"].values():
        use = arm["ok"] & np.isfinite(arm["dist_axis"]) & np.isfinite(arm["dist_hip"])
        far |= use & (arm["dist_axis"] >= 0.50)
        close |= use & (arm["dist_hip"] <= 0.25)
    if far.sum() < 1.0 * fps or close.sum() < 2.0 * fps:
        return []
    indexes = np.flatnonzero(close)
    return [
        Action(
            "hand_to_body",
            "Hand to the body",
            "both",
            int(indexes[0]),
            int(indexes[-1]),
            "A hand moves out from the torso and also rests against the hip.",
        )
    ]


def _hand_up(features: dict) -> list[Action]:
    """The wrist rises to shoulder height. There is no head point, so the face itself is not measured."""
    fps = features["fps"]
    short = max(4, int(round(0.30 * fps)))
    long = max(short + 1, int(round(0.80 * fps)))
    actions = []
    for side, arm in features["arms"].items():
        height = arm["height"]
        ok = arm["ok"]
        for width in range(short, long + 1, 2):
            for i, j in _windows(features["n"], width, step=2):
                if ok[i : j + 1].mean() < 0.75:
                    continue
                if not np.isfinite(height[i]) or not np.isfinite(height[j]):
                    continue
                if height[i] > 0.55 or height[j] < 0.95:
                    continue
                if height[j] - height[i] < 0.35:
                    continue
                actions.append(
                    Action(
                        "hand_up",
                        "Hand up",
                        side,
                        i,
                        j,
                        f"The {side} hand rises to about shoulder height.",
                    )
                )
    return actions
