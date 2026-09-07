"""Tests unitarios y secuencias sintéticas del detector HAR pose."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from har_pose.config import UNSUPPORTED_2D, DetectionConfig, KEYPOINTS
from har_pose.detectors import ActionAnalysisPipeline, ActionDetector, StateDetector
from har_pose.features import FeatureExtractor, PoseSequence
from har_pose.geometry import angle_at_joint
from har_pose.io_utils import seconds_to_timestamp, write_results_txt
from har_pose.temporal import SegmentProcessor, hysteresis_mask, mask_to_segments


def _base_pose() -> np.ndarray:
    """Pose frontal razonable en coords normalizadas [0,1], y_down."""
    # índices según KEYPOINTS
    p = np.zeros((8, 2), dtype=np.float64)
    p[0] = [0.40, 0.30]  # L shoulder
    p[1] = [0.60, 0.30]  # R shoulder
    p[2] = [0.35, 0.42]  # L elbow
    p[3] = [0.65, 0.42]  # R elbow
    p[4] = [0.33, 0.55]  # L wrist
    p[5] = [0.67, 0.55]  # R wrist
    p[6] = [0.43, 0.55]  # L hip
    p[7] = [0.57, 0.55]  # R hip
    return p


def _set_left_arm(pose: np.ndarray, elbow_angle_approx: str) -> np.ndarray:
    """Ajusta brazo izquierdo: 'flexed' ~90°, 'extended' ~170°."""
    p = pose.copy()
    sh = p[0]
    if elbow_angle_approx == "flexed":
        p[2] = sh + np.array([-0.02, 0.12])  # elbow down
        p[4] = sh + np.array([0.08, 0.10])  # wrist toward chest
    elif elbow_angle_approx == "extended":
        p[2] = sh + np.array([-0.08, 0.12])
        p[4] = sh + np.array([-0.16, 0.24])  # almost collinear down-left
    return p


class TestGeometry(unittest.TestCase):
    def test_angle_straight(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        c = np.array([[2.0, 0.0]])
        ang = angle_at_joint(a, b, c)
        self.assertTrue(np.isclose(ang[0], 180.0, atol=1e-4))

    def test_angle_right(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        # degenerate → nan
        c = np.array([[1.0, 0.0]])
        ang = angle_at_joint(a, b, c)
        self.assertTrue(np.isnan(ang[0]))

    def test_timestamp(self):
        self.assertEqual(seconds_to_timestamp(83.48), "00:01:23.480")
        self.assertEqual(seconds_to_timestamp(0), "00:00:00.000")


class TestHysteresisAndTemporal(unittest.TestCase):
    def test_no_flickering(self):
        # Ruido alrededor del threshold sin cruzar exit
        x = np.array([100, 160, 152, 158, 151, 159, 100], dtype=float)
        m = hysteresis_mask(x, enter=155, exit=140, mode="above")
        # Una vez entra en frame 1, no debe salir hasta <140
        self.assertTrue(m[1] and m[2] and m[3] and m[4] and m[5])
        self.assertFalse(m[0])
        self.assertFalse(m[6])

    def test_merge_gap(self):
        cfg = DetectionConfig(merge_gap_s=0.12, min_state_duration_s=0.05)
        fps = 25.0
        proc = SegmentProcessor(cfg, fps)
        mask = np.zeros(30, dtype=bool)
        mask[5:10] = True
        mask[12:18] = True  # gap de 2 frames = 0.08s < 0.12
        segs = mask_to_segments(mask, "TEST", 0)
        merged = proc.merge_gaps(segs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].start_frame, 5)
        self.assertEqual(merged[0].end_frame, 17)

    def test_min_duration_discard(self):
        cfg = DetectionConfig(min_state_duration_s=0.20, merge_gap_s=0.0)
        fps = 25.0
        proc = SegmentProcessor(cfg, fps)
        mask = np.zeros(30, dtype=bool)
        mask[5:7] = True  # 2 frames = 0.08s
        segs = proc.process(mask_to_segments(mask, "SHORT", 0))
        self.assertEqual(len(segs), 0)


class TestSyntheticActions(unittest.TestCase):
    def test_arm_stays_extended(self):
        fps = 25.0
        T = 40
        frames = []
        base = _base_pose()
        for _ in range(T):
            frames.append(_set_left_arm(base, "extended"))
        poses = np.stack(frames, axis=0)
        cfg = DetectionConfig(min_state_duration_s=0.15, smooth_window=3)
        segs, _ = ActionAnalysisPipeline(cfg).run(poses, fps)
        names = {s.name for s in segs}
        self.assertIn("ARM_EXTENDED_LEFT", names)
        # No debe inventar EXTEND si ya estaba extendido
        extend = [s for s in segs if s.name == "EXTEND_ARM_LEFT"]
        self.assertEqual(len(extend), 0)

    def test_flexed_to_extended_action(self):
        fps = 25.0
        frames = []
        base = _base_pose()
        # 15 flexed, 20 transitioning, 15 extended
        for i in range(50):
            if i < 15:
                frames.append(_set_left_arm(base, "flexed"))
            elif i < 35:
                a = _set_left_arm(base, "flexed")
                b = _set_left_arm(base, "extended")
                alpha = (i - 15) / 20.0
                frames.append((1 - alpha) * a + alpha * b)
            else:
                frames.append(_set_left_arm(base, "extended"))
        poses = np.stack(frames, axis=0)
        cfg = DetectionConfig(
            min_state_duration_s=0.12,
            min_action_duration_s=0.08,
            extend_delta_elbow_deg=20.0,
            smooth_window=3,
            action_min_frames=2,
        )
        segs, feats = ActionAnalysisPipeline(cfg).run(poses, fps)
        names = {s.name for s in segs}
        self.assertIn("ARM_EXTENDED_LEFT", names)
        self.assertIn("EXTEND_ARM_LEFT", names)

    def test_wrist_to_waist(self):
        fps = 25.0
        frames = []
        base = _base_pose()
        for i in range(40):
            p = base.copy()
            # muñeca izq empieza lejos y baja hacia cadera
            y0, y1 = 0.25, 0.54
            alpha = min(1.0, max(0.0, (i - 5) / 20.0))
            p[2] = [0.38, 0.40]
            p[4] = [0.42, (1 - alpha) * y0 + alpha * y1]
            frames.append(p)
        poses = np.stack(frames, axis=0)
        cfg = DetectionConfig(
            min_action_duration_s=0.08,
            approach_delta_norm=0.15,
            wrist_waist_enter_norm=0.55,
            wrist_waist_exit_norm=0.75,
            smooth_window=3,
            min_action_speed_norm_s=0.05,
        )
        segs, _ = ActionAnalysisPipeline(cfg).run(poses, fps)
        names = {s.name for s in segs}
        self.assertTrue(
            "WRIST_TO_WAIST_LEFT" in names or "WRIST_NEAR_WAIST_LEFT" in names
        )

    def test_red_classes_never_detected(self):
        fps = 25.0
        poses = np.stack([_base_pose() for _ in range(20)], axis=0)
        segs, _ = ActionAnalysisPipeline().run(poses, fps)
        names = {s.name for s in segs}
        for u in UNSUPPORTED_2D:
            self.assertNotIn(u, names)

    def test_write_txt_lists_unsupported(self):
        import tempfile

        cfg = DetectionConfig()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.txt"
            write_results_txt(path, [], source="x.npy", fps=25.0, config=cfg)
            text = path.read_text()
            self.assertIn("UNSUPPORTED WITH 2D KEYPOINTS", text)
            self.assertIn("ARM_FORWARD_DEPTH | CODE=2", text)


class TestKeypointConfig(unittest.TestCase):
    def test_keypoints_centralized(self):
        self.assertEqual(KEYPOINTS["left_shoulder"], 0)
        self.assertEqual(KEYPOINTS["right_hip"], 7)


if __name__ == "__main__":
    unittest.main()
