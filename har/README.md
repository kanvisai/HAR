# HAR

Marks seven gestures from a skeleton that is already computed. The clips are people walking a supermarket aisle. The program does not run pose estimation. It reads `poses_full.npy`.

## Gestures

Category 0 walks through, with free arms and sometimes a short spontaneous gesture. Category 1 stops to look and does not take a product. Category 5 keeps walking while carrying something. Distances are in torso lengths, from the shoulder center to the hip center. Looking down is not marked: the skeleton has no head point.

The same text is in `documentation/`. Notes from the example clips are in `gestures_0_1_5.txt`.

### walk

The hips travel along the aisle. Over about 0.9 seconds the hip center moves at least 0.45 torso lengths, and the body does not stay inside a tiny patch.

This is the travel shared by category 0 and category 5. Category 1 can still walk at the start or the end of the clip, outside the stop.

### stop

The hips stay in place. Over about 1.1 seconds the net move is at most 0.16 torso lengths and the body stays inside a radius of 0.15 torso lengths.

Frames already marked as a walk are left out, so a person who only walks through does not also get a stop. This is the pause in category 1.

### turn_torso

The shoulders go from facing the camera to a side view, or back the other way. The shoulder width, divided by the torso length, has to drop or rise by at least 0.20, between a front view (0.48 or more) and a side view (0.30 or less).

From this camera, turning toward the shelf is the side view. Category 1 does this while stopped. The window is about 0.45 to 1.20 seconds.

### arm_swing

While the person is walking, one wrist moves closer to its hip and then farther away, or the other way around. The spread is at least 0.16 torso lengths in about 0.8 seconds, and the distance changes direction.

This is the free arm of category 0. A hand that stays still does not count.

### carry

While the person is walking, one hand stays at nearly the same distance from its hip. Over about 1 second that distance varies by less than 0.07 torso lengths. The hand also sits away from the shoulder–hip line (at least 0.34 torso lengths) or up in front of the body (at least 0.42 of the way from the hip to the shoulder).

This is category 5: a basket, a bag or a product held while walking. A free swinging arm fails the steady-distance test.

### hand_up

One hand rises from below the chest to about shoulder height. The rise is at least 0.35 of the torso height and finishes at 0.95 or more, in about 0.3 to 0.8 seconds.

The skeleton has no head point, so this is the shoulder, not the face. It is the short spontaneous gesture seen in category 0.

### hand_to_body

A hand is at least 0.50 torso lengths out from the shoulder–hip line for 1 second or more, and a hand is also within 0.25 torso lengths of the hip for 2 seconds or more. The two do not have to be the same arm.

This is the pattern that separates category 6 from a walk-through, a stop to look, or someone already carrying a bag. When it is present, the clip is not called category 0, 1 or 5.

## Skeleton

`poses_full.npy` has shape `(frames, 8, 2)`. Coordinates run from 0 to 1. The y axis grows downward. A point that was not seen is NaN. It is not filled in and it is not drawn.

Points, in order, on the person's own left and right:

0. left shoulder
1. right shoulder
2. left elbow
3. right elbow
4. left wrist
5. right wrist
6. left hip
7. right hip

Distances are in torso lengths, from the shoulder center to the hip center. The usual layout is `videos/<category>/<clip>/user_<id>/poses_full.npy`, with `clip.mp4` and `meta.json` in the clip folder. The frame rate comes from `meta.json`.

## Setup

Python 3.12. Install inside a virtual environment. The system Python on Debian and Ubuntu refuses a direct `pip install`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r dependencies.txt
```

The text panel uses DejaVu Sans, which is not installed by pip:

```text
/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
```

On Debian or Ubuntu that file comes from the `fonts-dejavu-core` package.

Run the programs with `.venv/bin/python3`.

## Programs

Pass one skeleton file:

```bash
.venv/bin/python3 detect_json.py /path/to/poses_full.npy
.venv/bin/python3 visualize_ske.py /path/to/poses_full.npy
.venv/bin/python3 visualize_sup.py /path/to/poses_full.npy
```

`detect_json.py` writes `actions.json` next to the npy and a copy at `temp_output/actions.json`. It also writes `eval_cat_output.json` in both places.

`visualize_ske.py` opens the skeleton and writes `temp_output/skeleton.mp4` and `temp_output/eval_cat_output.json`.

`visualize_sup.py` opens the video with the skeleton on top. It also writes `actions.json`, `skeleton.mp4`, `overlay.mp4` and `eval_cat_output.json` into `temp_output`. The list on the right shows each action when it ends, with its time range. A row stays "In progress" while the action is still going.

`actions.json` lists the gestures. It does not name a category. The category is only in `eval_cat_output.json`.

Every run replaces the files in `temp_output`.

If no path is given, the programs use the first category-0 skeleton they find under the videos folder set in `har_actions/pose.py`.

## Category

`eval_cat_output.json` names one label: `c0`, `c1`, `c5`, `c100` or `inconclusive`. It records how long the skeleton was visible, how long each gesture lasted, and every rule in order. The first match wins. Later rules stay in the file with `reached` set to false.

A frame is visible when at least one keypoint is present. A gesture's time is the sum of its marked segments.

1. `hand_to_body` longer than 0 s → `inconclusive`. The gesture itself already requires a hand out from the torso for at least 1 s and a hand against the hip for at least 2 s. That pattern is not called `c0`, `c1` or `c5`.
2. Visible under 2.0 s → `inconclusive`.
3. `stop` at least 2.5 s and `walk` under 2.0 s → `c100`. The person stays still.
4. `stop` at least 2.5 s and `walk` at least 2.0 s → `c1`. The person walked and stopped.
5. `walk` at least 2.0 s and `carry` at least 2.0 s → `c5`.
6. `walk` at least 2.0 s → `c0`.
7. Anything shorter → `inconclusive`.

`turn_torso`, `arm_swing` and `hand_up` are measured and do not choose the category. Looking down is not measured. `c0` and `c5` may swap. A long stop is decided before `carry`, so a person who stops for at least 2.5 s is `c1` or `c100` even if they also carry something.

## Keys

Space pauses. The arrow keys, or A and D, move one frame. Q or Esc closes the window.

## Turning a gesture off

`actions_config.json` has an `enabled` flag for each id. Set it to `false`, save, and run the program again. All seven start enabled.
