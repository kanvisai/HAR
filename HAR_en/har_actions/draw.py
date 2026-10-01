"""Draw the skeleton and the action notices on a frame."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from har_actions.detect import Action
from har_actions.pose import BONES, NAMES, format_time

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

BONE_COLOR = (40, 210, 255)
POINT_COLOR = (0, 140, 255)
ACTIVE_COLOR = (40, 40, 255)
TEXT_COLOR = (245, 245, 245)

ACTION_COLORS = {
    "bring_to_chest": (0, 140, 255),
    "forearm_on_torso": (0, 200, 255),
    "hand_to_hand": (180, 40, 220),
    "cover_with_forearm": (40, 40, 230),
    "one_hand_hidden": (220, 180, 40),
    "lower_to_pocket": (200, 120, 20),
    "hand_at_pocket": (160, 80, 20),
    "hand_up_torso": (180, 60, 200),
    "arms_crossed": (40, 180, 40),
    "turn_torso": (200, 200, 200),
    "crouch": (20, 120, 200),
    "hand_behind_back": (180, 80, 160),
    "stiff_arm": (40, 220, 80),
}

SIDE_BONES = {
    "left": {(0, 2), (2, 4), (0, 6)},
    "right": {(1, 3), (3, 5), (1, 7)},
}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size)


def _pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def _bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _xy(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    return int(round(point[0] * width)), int(round(point[1] * height))


def active_sides(actions: list[Action]) -> set[str]:
    sides = set()
    for action in actions:
        if action.side == "both":
            sides.update(("left", "right"))
        else:
            sides.add(action.side)
    return sides


def draw_skeleton(
    image: np.ndarray,
    pose: np.ndarray,
    actions: list[Action] | None = None,
    radius: int = 6,
    thickness: int = 3,
) -> None:
    """Draw in normalized coordinates on a BGR image, in place."""
    height, width = image.shape[:2]
    active = active_sides(actions or [])
    active_bones = set()
    for side in active:
        active_bones |= SIDE_BONES.get(side, set())

    for a, b in BONES:
        if not (np.isfinite(pose[a]).all() and np.isfinite(pose[b]).all()):
            continue
        color = ACTIVE_COLOR if (a, b) in active_bones or (b, a) in active_bones else BONE_COLOR
        cv2.line(image, _xy(pose[a], width, height), _xy(pose[b], width, height), color, thickness, cv2.LINE_AA)

    for point in pose:
        if not np.isfinite(point).all():
            continue
        center = _xy(point, width, height)
        cv2.circle(image, center, radius, POINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(image, center, radius, (255, 255, 255), 1, cv2.LINE_AA)


def _time_bar(
    draw: ImageDraw.ImageDraw,
    all_actions: list[Action],
    frame: int,
    n_frames: int,
    x0: int,
    x1: int,
    y: int,
    height: int,
) -> None:
    draw.rectangle([x0, y, x1, y + height], fill=(30, 30, 34))
    if n_frames <= 1:
        return
    width = x1 - x0
    for action in all_actions:
        color = ACTION_COLORS.get(action.id, (255, 180, 0))
        color_rgb = (color[2], color[1], color[0])
        a = x0 + int(width * action.frame_start / n_frames)
        b = x0 + int(width * (action.frame_end + 1) / n_frames)
        draw.rectangle([a, y + 2, max(a + 2, b), y + height - 2], fill=color_rgb)
    cursor = x0 + int(width * frame / n_frames)
    draw.line([(cursor, y), (cursor, y + height)], fill=(255, 255, 255), width=2)


def compose(
    image: np.ndarray,
    pose: np.ndarray | None,
    frame_actions: list[Action],
    all_actions: list[Action],
    frame: int,
    n_frames: int,
    fps: float,
    title: str,
) -> np.ndarray:
    """Add the skeleton, the action notice and the timeline."""
    canvas = image.copy()
    if frame_actions:
        cv2.rectangle(canvas, (3, 3), (image.shape[1] - 4, image.shape[0] - 4), (40, 40, 220), 6)
    if pose is not None:
        draw_skeleton(canvas, pose, frame_actions)

    height, width = canvas.shape[:2]
    band = 118
    strip = np.zeros((band, width, 3), dtype=np.uint8)
    strip[:] = (16, 16, 20)
    if frame_actions:
        strip[:, :10] = (40, 40, 220)
    output = np.vstack([strip, canvas])

    image_pil = _pil(output)
    draw = ImageDraw.Draw(image_pil)
    font = _font(22, bold=True)
    small = _font(18)
    clock = format_time(frame / fps)
    draw.text((18, 10), f"{title}    {clock}    frame {frame}", font=font, fill=(235, 235, 235))

    if frame_actions:
        text = "  ·  ".join(f"{a.name} ({a.side})" for a in frame_actions[:3])
        draw.text((18, 46), "ACTION  " + text, font=small, fill=(255, 210, 80))
        if len(frame_actions) > 3:
            extra = "  ·  ".join(f"{a.name} ({a.side})" for a in frame_actions[3:6])
            draw.text((18, 74), extra, font=small, fill=(255, 210, 80))
    else:
        draw.text((18, 46), "No concealment gesture", font=small, fill=(170, 170, 170))

    bar_y = height + band - 22
    _time_bar(draw, all_actions, frame, n_frames, 18, width - 18, bar_y, 14)
    return _bgr(image_pil)


def person_crop(image: np.ndarray, pose: np.ndarray, margin: int = 70) -> np.ndarray | None:
    height, width = image.shape[:2]
    valid = pose[np.isfinite(pose).all(axis=1)]
    if len(valid) < 4:
        return None
    xs = valid[:, 0] * width
    ys = valid[:, 1] * height
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(width, int(xs.max()) + margin)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(height, int(ys.max()) + margin)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return image[y0:y1, x0:x1]


def place_crop(image: np.ndarray, crop: np.ndarray, box_width: int = 280) -> None:
    if crop is None or crop.size == 0:
        return
    h, w = crop.shape[:2]
    box_height = int(box_width * h / w)
    box = cv2.resize(crop, (box_width, box_height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(box, (0, 0), (box_width - 1, box_height - 1), (255, 255, 255), 2)
    y0, x0 = 8, image.shape[1] - box_width - 12
    if y0 + box_height > image.shape[0] or x0 < 0:
        return
    image[y0 : y0 + box_height, x0 : x0 + box_width] = box


def skeleton_canvas(pose: np.ndarray, width: int = 960, height: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Black background with the skeleton zoomed to the body in this frame."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (18, 18, 22)
    valid = pose[np.isfinite(pose).all(axis=1)]
    if len(valid) < 2:
        return image, pose
    x0, y0 = valid.min(axis=0)
    x1, y1 = valid.max(axis=0)
    dx = max(float(x1 - x0), 0.05)
    dy = max(float(y1 - y0), 0.05)
    margin = 0.08
    x0 -= dx * margin
    y0 -= dy * margin
    dx *= 1 + 2 * margin
    dy *= 1 + 2 * margin
    scale = min((width - 40) / dx, (height - 40) / dy)
    shifted = pose.copy()
    shifted[:, 0] = (pose[:, 0] - x0) * scale + 20
    shifted[:, 1] = (pose[:, 1] - y0) * scale + 20
    for a, b in BONES:
        if not (np.isfinite(shifted[a]).all() and np.isfinite(shifted[b]).all()):
            continue
        p1 = (int(shifted[a, 0]), int(shifted[a, 1]))
        p2 = (int(shifted[b, 0]), int(shifted[b, 1]))
        cv2.line(image, p1, p2, BONE_COLOR, 4, cv2.LINE_AA)
    for index, point in enumerate(shifted):
        if not np.isfinite(point).all():
            continue
        center = (int(point[0]), int(point[1]))
        cv2.circle(image, center, 8, POINT_COLOR, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            NAMES[index].replace("_", " "),
            (center[0] + 10, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return image, shifted


def recolor_active_skeleton(image: np.ndarray, shifted: np.ndarray, actions: list[Action]) -> None:
    active = active_sides(actions)
    active_bones = set()
    for side in active:
        active_bones |= SIDE_BONES.get(side, set())
    for a, b in BONES:
        if (a, b) not in active_bones and (b, a) not in active_bones:
            continue
        if not (np.isfinite(shifted[a]).all() and np.isfinite(shifted[b]).all()):
            continue
        p1 = (int(shifted[a, 0]), int(shifted[a, 1]))
        p2 = (int(shifted[b, 0]), int(shifted[b, 1]))
        cv2.line(image, p1, p2, ACTIVE_COLOR, 6, cv2.LINE_AA)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def join_list(image: np.ndarray, actions: list[Action], frame: int, fps: float) -> np.ndarray:
    """Place on the right the list of finished actions, with their time range."""
    max_height = 720
    height, width = image.shape[:2]
    if height > max_height:
        scale = max_height / height
        image = cv2.resize(image, (int(width * scale), max_height), interpolation=cv2.INTER_AREA)
    ongoing = [a for a in actions if a.frame_start <= frame <= a.frame_end]
    if ongoing:
        h, w = image.shape[:2]
        cv2.rectangle(image, (3, 3), (w - 4, h - 4), (40, 40, 220), 4)

    clock = format_time(frame / fps)
    marked = _pil(image)
    drawing = ImageDraw.Draw(marked)
    image_height = image.shape[0]
    drawing.rectangle([8, image_height - 42, 250, image_height - 8], fill=(0, 0, 0))
    drawing.text((14, image_height - 38), f"{clock}   frame {frame}", font=_font(20, bold=True), fill=(255, 255, 255))
    image = _bgr(marked)

    panel = _action_panel(actions, frame, fps, image.shape[0], 480)
    return np.hstack([image, panel])


def _action_panel(actions: list[Action], frame: int, fps: float, height: int, width: int) -> np.ndarray:
    ongoing = [a for a in actions if a.frame_start <= frame < a.frame_end]
    done = [a for a in actions if frame >= a.frame_end]
    done.sort(key=lambda a: (a.frame_end, a.frame_start, a.name))

    canvas = Image.new("RGB", (width, max(height, 1)), (16, 16, 20))
    draw = ImageDraw.Draw(canvas)
    title = _font(22, bold=True)
    time_font = _font(18, bold=True)
    body = _font(17)
    small = _font(15)
    margin = 16
    usable = width - 2 * margin

    draw.text((margin, 14), "Actions", font=title, fill=(245, 245, 245))
    y = 52
    if ongoing:
        draw.text((margin, y), "In progress", font=small, fill=(255, 196, 80))
        y += 24
        for action in ongoing:
            for line in _wrap(draw, f"{action.name} ({action.side})", body, usable):
                draw.text((margin, y), line, font=body, fill=(230, 230, 230))
                y += 22
            y += 8
        draw.line([(margin, y), (width - margin, y)], fill=(55, 55, 62), width=1)
        y += 12

    if not done:
        for line in _wrap(draw, "When an action ends, it appears here with its time range.", small, usable):
            draw.text((margin, y), line, font=small, fill=(150, 150, 150))
            y += 20
        return _bgr(canvas)

    blocks: list[list[tuple[str, ImageFont.ImageFont, tuple[int, int, int]]]] = []
    for action in done:
        data = action.to_dict(fps)
        lines = [(f"{data['t_start']} – {data['t_end']}", time_font, (255, 210, 90))]
        for line in _wrap(draw, action.name, body, usable):
            lines.append((line, body, (235, 235, 235)))
        lines.append((action.side, small, (170, 170, 170)))
        blocks.append(lines)

    heights = []
    for lines in blocks:
        block_height = 8 + sum(22 if font != small else 20 for _, font, _ in lines) + 10
        heights.append(block_height)
    total = sum(heights)
    available = height - y - 8
    offset = 0 if total <= available else total - available

    crop = Image.new("RGB", (width, max(available, 1)), (16, 16, 20))
    drawing = ImageDraw.Draw(crop)
    cursor = -offset
    for lines, block_height in zip(blocks, heights):
        if cursor + block_height >= 0 and cursor < available:
            yy = cursor + 4
            for text, font, color in lines:
                if 0 <= yy < available - 4:
                    drawing.text((margin, yy), text, font=font, fill=color)
                yy += 22 if font != small else 20
            line_y = cursor + block_height - 6
            if 0 <= line_y < available:
                drawing.line([(margin, line_y), (width - margin, line_y)], fill=(48, 48, 56), width=1)
        cursor += block_height
    canvas.paste(crop, (0, y))
    return _bgr(canvas)


def play(title: str, n: int, fps: float, build) -> None:
    """Open a window. Space pauses, arrows move one frame, Q quits."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    index = 0
    paused = False
    opened = False
    while True:
        if opened and cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.imshow(title, build(index))
        opened = True
        wait = 30 if paused or index >= n - 1 else max(1, int(round(1000 / fps)))
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key in (81, 2, ord("a")):
            index = max(0, index - 1)
            paused = True
            continue
        if key in (83, 3, ord("d")):
            index = min(n - 1, index + 1)
            paused = True
            continue
        if not paused and key == 255 and index < n - 1:
            index += 1
    cv2.destroyAllWindows()


def save_temp(path: Path, fps: float, n: int, build) -> None:
    """Write the video to path, replacing the file if it already existed."""
    if n <= 0:
        return
    first = build(0)
    height, width = first.shape[:2]
    writer = open_writer(path, fps, width, height)
    writer.write(first)
    for index in range(1, n):
        writer.write(build(index))
    writer.release()


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create the video {path}")
    return writer
