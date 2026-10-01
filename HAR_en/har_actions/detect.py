"""
Concealment gestures from the skeleton.

Only what happens after the product is already in hand is considered:
bringing it to the body, covering it, lowering it to the hip, or carrying it close while walking.
With 8 points the object is not visible: each notice is a gesture consistent with that.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from har_actions.pose import (
    ARM_SIDE,
    HIP_L,
    HIP_R,
    SHOULDER_L,
    SHOULDER_R,
    format_time,
)

ACTION_CONFIG = Path(__file__).resolve().parents[1] / "actions_config.json"

# These items from the original list cannot be seen without the object or the bag.
NOT_OBSERVABLE = (
    "Open the backpack, bag, or cart",
    "Put an arm into a bag and pull it out empty",
    "Cover the product with another item",
)


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


def detect(poses: np.ndarray, fps: float) -> list[Action]:
    """Smoothed poses, shape (T, 8, 2)."""
    features = _features(poses, fps)
    actions: list[Action] = []
    actions += _bring_to_torso(features)
    actions += _forearm_tucked(features)
    actions += _hands_together(features)
    actions += _cover_with_forearm(features)
    actions += _one_hand_hidden(features)
    actions += _lower_to_hip(features)
    actions += _up_the_torso(features)
    actions += _arms_crossed(features)
    actions += _turn_torso(features)
    actions += _crouch(features)
    actions += _hand_behind(features)
    actions += _stiff_arm(features)
    actions = _merge(actions, gap=4)
    actions += _stay_after_lowering(features, actions)
    config = load_action_config()
    actions = [action for action in actions if action_enabled(action.id, config)]
    actions.sort(key=lambda a: (a.frame_start, a.frame_end, a.id, a.side))
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
        return "No concealment gesture was marked in this clip."
    lines = [f"{len(actions)} concealment gesture(s):", ""]
    for action in actions:
        data = action.to_dict(fps)
        lines.append(
            f"  {data['t_start']} → {data['t_end']}   "
            f"{action.name}   [{action.side}]"
        )
        lines.append(f"      {action.detail}")
        lines.append(
            f"      frames {action.frame_start}–{action.frame_end}  "
            f"({data['duration_s']} s)"
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def _features(poses: np.ndarray, fps: float) -> dict:
    p = poses
    t = len(p)
    shoulder_c = 0.5 * (p[:, SHOULDER_L] + p[:, SHOULDER_R])
    hip_c = 0.5 * (p[:, HIP_L] + p[:, HIP_R])
    torso = np.linalg.norm(shoulder_c - hip_c, axis=1)
    width = np.linalg.norm(p[:, SHOULDER_L] - p[:, SHOULDER_R], axis=1)
    torso_ok = (torso > 0.06) & (torso < 0.55) & (hip_c[:, 1] > shoulder_c[:, 1] - 0.02)
    scale = np.maximum(torso, 1e-3)

    chest = 0.50 * shoulder_c + 0.50 * hip_c
    abdomen = 0.28 * shoulder_c + 0.72 * hip_c

    speed = np.zeros(t)
    if t > 1:
        step = np.linalg.norm(np.diff(hip_c, axis=0), axis=1) * fps
        speed[1:] = step / scale[1:]
        speed[0] = speed[1]

    arms = {}
    for side, (shoulder, elbow, wrist, hip) in ARM_SIDE.items():
        w = p[:, wrist]
        s = p[:, shoulder]
        e = p[:, elbow]
        hip_pt = p[:, hip]
        upper = np.linalg.norm(e - s, axis=1)
        forearm = np.linalg.norm(w - e, axis=1)
        angle = _angle(s, e, w)
        # An elbow under 40° is a tracker collapse, not a gesture.
        ok = torso_ok & (upper > 0.10 * scale) & (forearm > 0.08 * scale) & (angle > 40)
        torso_height = np.maximum(hip_pt[:, 1] - s[:, 1], 1e-3)
        # 0 = hip height, 1 = shoulder height. Above the shoulder is > 1.
        height = (hip_pt[:, 1] - w[:, 1]) / torso_height
        arms[side] = {
            "ok": ok,
            "angle": angle,
            "dist_chest": np.linalg.norm(w - chest, axis=1) / scale,
            "dist_abdomen": np.linalg.norm(w - abdomen, axis=1) / scale,
            "dist_hip": np.linalg.norm(w - hip_pt, axis=1) / scale,
            "dist_axis": _segment_distance(w, s, hip_pt) / scale,
            "height": height,
            "crosses": (w[:, 0] - shoulder_c[:, 0]) / np.maximum(width, 1e-3),
            "wrist": w,
            "hip": hip_pt,
        }
        if side == "left":
            arms[side]["crosses_inward"] = arms[side]["crosses"]
        else:
            arms[side]["crosses_inward"] = -arms[side]["crosses"]

    wrists = np.linalg.norm(p[:, 4] - p[:, 5], axis=1) / scale
    return {
        "fps": fps,
        "n": t,
        "torso_ok": torso_ok,
        "scale": scale,
        "width": width,
        "frontal_ratio": width / scale,
        "speed": speed,
        "hip_c": hip_c,
        "shoulder_c": shoulder_c,
        "arms": arms,
        "wrist_sep": wrists,
        "poses": p,
    }


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    cos = np.sum(ba * bc, axis=1) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.sum(ab * ab, axis=1) + 1e-8
    t = np.clip(np.sum((p - a) * ab, axis=1) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)


def _spans(mask: np.ndarray, minimum: int, gap: int = 2) -> list[tuple[int, int]]:
    """Inclusive spans. Short gaps are closed and brief spans are dropped."""
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
        group.sort(key=lambda a: a.frame_start)
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


def _bring_to_torso(features: dict) -> list[Action]:
    """The wrist drops toward the abdomen. A straight reach to the shelf does not count."""
    actions = []
    width = max(6, int(round(0.70 * features["fps"])))
    for side, arm in features["arms"].items():
        dist = arm["dist_abdomen"]
        for i, j in _windows(features["n"], width, step=2):
            if arm["ok"][i : j + 1].mean() < 0.8:
                continue
            if arm["height"][i] < 0.45 or arm["height"][j] > 0.72:
                continue
            if arm["height"][j] > arm["height"][i] - 0.12:
                continue
            drop = dist[i] - dist[j]
            if drop < 0.22 or dist[j] > 0.48 or dist[i] < 0.60:
                continue
            if not (0.22 <= arm["height"][j] <= 0.72):
                continue
            actions.append(
                Action(
                    "bring_to_chest",
                    "Bring the product to the chest or abdomen",
                    side,
                    i,
                    j,
                    (
                        f"The {side} wrist moves toward the abdomen: "
                        f"from {dist[i]:.2f} to {dist[j]:.2f} torso lengths."
                    ),
                )
            )
    return actions


def _forearm_tucked(features: dict) -> list[Action]:
    minimum = max(5, int(round(0.45 * features["fps"])))
    actions = []
    for side, arm in features["arms"].items():
        mask = (
            arm["ok"]
            & (arm["angle"] > 55)
            & (arm["angle"] < 125)
            & (arm["dist_axis"] < 0.20)
            & (arm["height"] > 0.18)
            & (arm["height"] < 0.62)
        )
        for i, j in _spans(mask, minimum):
            actions.append(
                Action(
                    "forearm_on_torso",
                    "Press the forearm against the torso",
                    side,
                    i,
                    j,
                    (
                        f"The {side} elbow is bent and the {side} wrist "
                        f"stays against the torso axis."
                    ),
                )
            )
    return actions


def _hands_together(features: dict) -> list[Action]:
    """Both wrists meet in front of the abdomen: passing the object from hand to hand."""
    left, right = features["arms"]["left"], features["arms"]["right"]
    minimum = max(6, int(round(0.50 * features["fps"])))
    smaller_angle = np.minimum(left["angle"], right["angle"])
    mask = (
        left["ok"]
        & right["ok"]
        & (features["wrist_sep"] < 0.28)
        & (features["frontal_ratio"] > 0.20)
        & (smaller_angle > 50)
        & (smaller_angle < 145)
        & (left["height"] > 0.14)
        & (left["height"] < 0.62)
        & (right["height"] > 0.14)
        & (right["height"] < 0.62)
    )
    actions = []
    for i, j in _spans(mask, minimum):
        actions.append(
            Action(
                "hand_to_hand",
                "Pass the object from one hand to the other",
                "both",
                i,
                j,
                "Both wrists meet in front of the torso, with at least one elbow bent.",
            )
        )
    return actions


def _cover_with_forearm(features: dict) -> list[Action]:
    minimum = max(4, int(round(0.28 * features["fps"])))
    actions = []
    for side, arm in features["arms"].items():
        # The torso must be somewhat facing the camera: in a pure profile the midline is not visible.
        mask = (
            arm["ok"]
            & (features["frontal_ratio"] > 0.22)
            & (arm["crosses_inward"] > 0.45)
            & (arm["angle"] < 155)
            & (arm["height"] > 0.20)
            & (arm["height"] < 0.95)
        )
        for i, j in _spans(mask, minimum):
            actions.append(
                Action(
                    "cover_with_forearm",
                    "Cover the product with the forearm",
                    side,
                    i,
                    j,
                    f"The {side} wrist crosses in front of the chest midline.",
                )
            )
    return actions


def _one_hand_hidden(features: dict) -> list[Action]:
    """One hand stays out, at mid height; the other stays low, by the hip."""
    minimum = max(4, int(round(0.35 * features["fps"])))
    actions = []
    pairs = (("left", "right"), ("right", "left"))
    for hidden, visible in pairs:
        low = features["arms"][hidden]
        high = features["arms"][visible]
        mask = (
            low["ok"]
            & high["ok"]
            & (low["dist_hip"] < 0.38)
            & (low["height"] < 0.35)
            & (high["dist_axis"] > 0.55)
            & (high["height"] > 0.35)
            & (high["height"] < 0.90)
            & (high["angle"] > 140)
        )
        for i, j in _spans(mask, minimum):
            actions.append(
                Action(
                    "one_hand_hidden",
                    "One hand in view and the other hidden against the body",
                    hidden,
                    i,
                    j,
                    (
                        f"The {hidden} hand stays by the hip while the {visible} hand "
                        f"stays away from the torso, at mid height."
                    ),
                )
            )
    return actions


def _lower_to_hip(features: dict) -> list[Action]:
    """Drops from above (just after the grab) to the hip. An arm that was already hanging does not count."""
    actions = []
    width = max(8, int(round(0.85 * features["fps"])))
    for side, arm in features["arms"].items():
        drops = []
        for i, j in _windows(features["n"], width, step=2):
            if i < 3 or arm["ok"][i : j + 1].mean() < 0.75:
                continue
            if arm["height"][i] < 0.75 or np.median(arm["height"][i - 3 : i + 1]) < 0.72:
                continue
            mid = (i + j) // 2
            if not (arm["height"][i] > arm["height"][mid] > arm["height"][j]):
                continue
            fall = arm["height"][i] - arm["height"][j]
            if fall < 0.40:
                continue
            if arm["height"][j] > 0.32 or arm["dist_hip"][j] > 0.48:
                continue
            drops.append((i, j, fall))
        for i, j, fall in drops:
            actions.append(
                Action(
                    "lower_to_pocket",
                    "Lower the hand toward the pocket",
                    side,
                    i,
                    j,
                    (
                        f"The {side} hand drops from above down to the hip "
                        f"(relative height falls {fall:.2f})."
                    ),
                )
            )
        actions += _pocket_entry(features, side, arm)
    return actions


def _pocket_entry(features: dict, side: str, arm: dict) -> list[Action]:
    """The hand drops from the chest to the pocket in a short time and stays there.

    The tracker often collapses the elbow right at the entry, so intermediate
    frames are not required to be valid.
    """
    fps = features["fps"]
    n = features["n"]
    short = max(3, int(round(0.24 * fps)))
    # Up to ~1 s: as the hand goes in, the forearm shrinks and those frames are not valid.
    long = max(short + 1, int(round(1.05 * fps)))
    stay = max(4, int(round(0.40 * fps)))
    actions = []
    height = arm["height"]
    dist = arm["dist_hip"]
    ok = arm["ok"]
    for j in range(long, n):
        if not (ok[j] and height[j] <= 0.22 and dist[j] <= 0.32):
            continue
        if height[j - 1] <= 0.22 and dist[j - 1] <= 0.32:
            continue
        start = None
        for i in range(j - short, max(2, j - long) - 1, -1):
            if i < 3 or not ok[i] or height[i] < 0.55:
                continue
            if float(np.nanmedian(height[i - 3 : i + 1])) < 0.50:
                continue
            if height[i] - height[j] < 0.35:
                continue
            start = i
            break
        if start is None:
            continue
        end = min(n - 1, j + stay)
        low = sum(height[k] < 0.30 and dist[k] < 0.42 for k in range(j, end + 1))
        if low < stay:
            continue
        fall = height[start] - height[j]
        actions.append(
            Action(
                "lower_to_pocket",
                "Lower the hand toward the pocket",
                side,
                start,
                j,
                (
                    f"The {side} hand moves from the chest to the hip "
                    f"and stays there (relative height falls {fall:.2f})."
                ),
            )
        )
    return actions


def _stay_after_lowering(features: dict, actions: list[Action]) -> list[Action]:
    """The hand stays at the hip only right after a drop, and only for a short time."""
    cap = max(6, int(round(0.70 * features["fps"])))
    minimum = max(4, int(round(0.35 * features["fps"])))
    output = []
    for action in actions:
        if action.id != "lower_to_pocket":
            continue
        arm = features["arms"][action.side]
        j = action.frame_end
        k = j
        limit = min(features["n"] - 1, j + cap)
        while k < limit and arm["ok"][k] and arm["dist_hip"][k] < 0.40 and arm["height"][k] < 0.32:
            k += 1
        if (k - j) >= minimum:
            output.append(
                Action(
                    "hand_at_pocket",
                    "Keep the hand in the pocket or at the hip",
                    action.side,
                    j,
                    k,
                    f"After dropping, the {action.side} wrist stays by the hip.",
                )
            )
    return output


def _up_the_torso(features: dict) -> list[Action]:
    """Rises against the body: the hem of the clothes, or the hand under the garment. Not a reach to the shelf."""
    actions = []
    width = max(5, int(round(0.55 * features["fps"])))
    for side, arm in features["arms"].items():
        for i, j in _windows(features["n"], width, step=2):
            if not arm["ok"][i : j + 1].mean() > 0.8:
                continue
            if arm["height"][i] > 0.40:
                continue
            rise = arm["height"][j] - arm["height"][i]
            if rise < 0.30 or not (0.45 <= arm["height"][j] <= 0.78):
                continue
            if np.median(arm["dist_axis"][i : j + 1]) > 0.22:
                continue
            if np.median(arm["angle"][i : j + 1]) > 140:
                continue
            actions.append(
                Action(
                    "hand_up_torso",
                    "Slide the hand up the torso, against the body",
                    side,
                    i,
                    j,
                    (
                        f"The {side} wrist rises from the hip toward the chest "
                        f"without leaving the torso."
                    ),
                )
            )
    return actions


def _arms_crossed(features: dict) -> list[Action]:
    left, right = features["arms"]["left"], features["arms"]["right"]
    minimum = max(4, int(round(0.30 * features["fps"])))
    mask = (
        left["ok"]
        & right["ok"]
        & (features["frontal_ratio"] > 0.20)
        & (left["crosses_inward"] > 0.25)
        & (right["crosses_inward"] > 0.25)
        & (left["angle"] < 155)
        & (right["angle"] < 155)
        & (left["height"] > 0.30)
        & (left["height"] < 1.05)
        & (right["height"] > 0.30)
        & (right["height"] < 1.05)
    )
    actions = []
    for i, j in _spans(mask, minimum):
        actions.append(
            Action(
                "arms_crossed",
                "Cross the arms over the chest",
                "both",
                i,
                j,
                "Each wrist crosses to the opposite side of the chest, with the elbows bent.",
            )
        )
    return actions


def _turn_torso(features: dict) -> list[Action]:
    """Shoulder width narrows: the body turns sideways and one hand stays close to the torso."""
    ratio = features["frontal_ratio"].copy()
    ok = features["torso_ok"]
    actions = []
    width = max(6, int(round(0.70 * features["fps"])))
    for i, j in _windows(features["n"], width, step=2):
        if ok[i : j + 1].mean() < 0.8:
            continue
        if ratio[i] < 0.48 or ratio[j] > 0.22:
            continue
        if ratio[i] - ratio[j] < 0.26:
            continue
        after = ratio[j : min(features["n"], j + 8)]
        if len(after) < 5 or np.median(after) > 0.26:
            continue
        hand_close = False
        for arm in features["arms"].values():
            if arm["ok"][i : j + 1].mean() > 0.6 and np.median(arm["dist_axis"][i : j + 1]) < 0.35:
                hand_close = True
        if not hand_close:
            continue
        actions.append(
            Action(
                "turn_torso",
                "Turn the torso sideways to hide the hands",
                "both",
                i,
                j,
                (
                    f"The shoulders go from facing the camera (width {ratio[i]:.2f}) "
                    f"to a side view (width {ratio[j]:.2f}), with one hand close to the body."
                ),
            )
        )
    return actions


def _crouch(features: dict) -> list[Action]:
    """The hips drop in the image and both hands end at leg height."""
    y = features["hip_c"][:, 1]
    left, right = features["arms"]["left"], features["arms"]["right"]
    actions = []
    width = max(5, int(round(0.70 * features["fps"])))
    for i, j in _windows(features["n"], width, step=2):
        if features["torso_ok"][i : j + 1].mean() < 0.8:
            continue
        drop = y[j] - y[i]
        if drop < 0.045:
            continue
        hands_low = (
            left["ok"][j]
            and right["ok"][j]
            and left["height"][j] < 0.25
            and right["height"][j] < 0.25
        )
        if not hands_low:
            continue
        actions.append(
            Action(
                "crouch",
                "Crouch with the hands at leg height",
                "both",
                i,
                j,
                "The body drops in the image and both wrists end at hip height or below.",
            )
        )
    return actions


def _hand_behind(features: dict) -> list[Action]:
    minimum = max(4, int(round(0.30 * features["fps"])))
    actions = []
    pairs = (
        ("left", "right"),
        ("right", "left"),
    )
    for side, opposite in pairs:
        arm = features["arms"][side]
        opposite_hip = features["arms"][opposite]["hip"]
        dist_opposite = np.linalg.norm(arm["wrist"] - opposite_hip, axis=1) / features["scale"]
        mask = (
            arm["ok"]
            & (features["frontal_ratio"] > 0.28)
            & (dist_opposite < arm["dist_hip"] - 0.15)
            & (dist_opposite < 0.40)
            & (arm["height"] < 0.45)
            & (arm["angle"] < 150)
            & (arm["angle"] > 50)
        )
        for i, j in _spans(mask, minimum):
            actions.append(
                Action(
                    "hand_behind_back",
                    "Move one hand to the back or to the opposite hip",
                    side,
                    i,
                    j,
                    f"The {side} wrist is closer to the opposite hip than to its own.",
                )
            )
    return actions


def _stiff_arm(features: dict) -> list[Action]:
    """The person is walking and one arm does not swing: it stays at the same distance from its hip."""
    width = max(8, int(round(1.0 * features["fps"])))
    actions = []
    sides = ("left", "right")
    for i, j in _windows(features["n"], width, step=3):
        if features["torso_ok"][i : j + 1].mean() < 0.85:
            continue
        if np.median(features["speed"][i : j + 1]) < 0.10:
            continue
        deviation = {}
        for side in sides:
            arm = features["arms"][side]
            if arm["ok"][i : j + 1].mean() < 0.85:
                deviation[side] = None
                continue
            rel = arm["wrist"][i : j + 1] - arm["hip"][i : j + 1]
            rel = rel / features["scale"][i : j + 1, None]
            deviation[side] = float(np.mean(np.linalg.norm(rel - rel.mean(axis=0), axis=1)))
        for side, other in (sides, sides[::-1]):
            if deviation[side] is None or deviation[other] is None:
                continue
            arm = features["arms"][side]
            height = float(np.median(arm["height"][i : j + 1]))
            close = float(np.median(arm["dist_hip"][i : j + 1])) < 0.40
            # Against the torso, not a dead arm hanging by the thigh.
            on_torso = 0.18 <= height <= 0.55
            if deviation[side] < 0.05 and deviation[other] > 0.12 and close and on_torso:
                actions.append(
                    Action(
                        "stiff_arm",
                        "Walk with one arm held against the body, without swinging",
                        side,
                        i,
                        j,
                        (
                            f"While the body is moving, the {side} arm barely moves "
                            f"relative to its hip and the other arm swings."
                        ),
                    )
                )
    return actions
