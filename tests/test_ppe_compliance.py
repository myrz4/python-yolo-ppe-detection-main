import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "06_roi_telegram_logging_ppe_detection.py"
spec = importlib.util.spec_from_file_location("ppe_detection", SCRIPT)
ppe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppe)


class Box:
    def __init__(self, class_id, bbox):
        self.cls = [class_id]
        self.conf = [0.9]
        self.xyxy = [SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(bbox)))]


class ComplianceDisplayTests(unittest.TestCase):
    def run_frame(self, people, mode=1, tracking=False, required_ppe=None):
        names = [ppe.PERSON_CLASS, *ppe.PPE_PROPER, *ppe.PPE_VIOLATIONS]
        boxes = []
        for index, items in enumerate(people):
            x = index * 200
            boxes.append(Box(0, [x, 0, x + 150, 200]))
            for item in items:
                boxes.append(Box(names.index(item), [x + 20, 20, x + 40, 40]))

        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        result = SimpleNamespace(boxes=boxes, plot=lambda: frame.copy())
        person_ids = list(range(19, 19 + len(people))) if tracking else list(range(len(people)))
        tracker = SimpleNamespace(update=lambda bboxes: person_ids) if tracking else None
        app = SimpleNamespace(
            cap=SimpleNamespace(read=lambda: (True, frame.copy())),
            frame_count=0,
            use_roi=False,
            model=SimpleNamespace(names=names, predict=lambda **kwargs: [result]),
            violation_mode=mode,
            required_ppe=list(ppe.REQUIRED_PPE if required_ppe is None else required_ppe),
            person_tracker=tracker,
            logged_person_ids=set(person_ids),
            video_label=Mock(),
            status_label=Mock(),
            play_status_sound=Mock(),
        )
        app._check_proximity = lambda a, b: ppe.PPEDetectionApp._check_proximity(app, a, b)
        app.check_violations = lambda results: ppe.PPEDetectionApp.check_violations(app, results)

        with patch.object(ppe.cv2, "putText") as put_text, \
             patch.object(ppe, "QImage"), patch.object(ppe, "QPixmap"):
            ppe.PPEDetectionApp.update_frame(app)
        expected_sound = "VIOLATION" if "VIOLATION" in app.status_label.setText.call_args.args[0] else (
            "PASS" if people else None
        )
        app.play_status_sound.assert_called_once_with(expected_sound)
        labels = [call.args[1] for call in put_text.call_args_list]
        return app.status_label.setText.call_args.args[0], labels

    def test_full_ppe_shows_pass(self):
        status, labels = self.run_frame([ppe.PPE_PROPER])
        self.assertEqual(status, "Status: PASS")
        self.assertEqual(labels, ["PASS", "PASS"])

    def test_missing_ppe_shows_violation(self):
        status, labels = self.run_frame([[item for item in ppe.PPE_PROPER if item != "boots"]])
        self.assertIn("VIOLATION", status)
        self.assertIn("boots", status)
        self.assertEqual(labels, ["VIOLATION", "VIOLATION"])

    def test_explicit_missing_ppe_overrides_positive_detection(self):
        status, labels = self.run_frame([ppe.PPE_PROPER + ["no_helmet"]])
        self.assertIn("VIOLATION", status)
        self.assertEqual(labels, ["VIOLATION", "VIOLATION"])

    def test_mixed_people_show_individual_and_overall_status(self):
        status, labels = self.run_frame([ppe.PPE_PROPER, ["helmet"]])
        self.assertIn("VIOLATION", status)
        self.assertEqual(labels, ["PASS", "VIOLATION", "VIOLATION"])

    def test_tracked_people_show_only_ids_and_overall_status(self):
        status, labels = self.run_frame([ppe.PPE_PROPER, ["helmet"]], tracking=True)
        self.assertIn("VIOLATION", status)
        self.assertEqual(labels, ["ID:19", "ID:20", "VIOLATION"])

    def test_tracked_compliant_person_shows_only_id(self):
        status, labels = self.run_frame([ppe.PPE_PROPER], tracking=True)
        self.assertEqual(status, "Status: PASS")
        self.assertEqual(labels, ["ID:19", "PASS"])

    def test_no_person_has_no_pass_or_violation(self):
        status, labels = self.run_frame([])
        self.assertIn("No person detected", status)
        self.assertEqual(labels, [])

    def test_required_ppe_mode_preserves_its_rules(self):
        status, labels = self.run_frame([ppe.REQUIRED_PPE], mode=2)
        self.assertEqual(status, "Status: PASS")
        self.assertEqual(labels, ["PASS", "PASS"])

    def test_mode2_uses_the_selected_required_ppe(self):
        status, _ = self.run_frame(
            [["helmet", "vest"]], mode=2, required_ppe=["helmet"]
        )
        self.assertEqual(status, "Status: PASS")

        status, _ = self.run_frame(
            [["helmet", "no_boots"]], mode=2, required_ppe=["helmet"]
        )
        self.assertEqual(status, "Status: PASS")

        status, _ = self.run_frame(
            [["vest", "no_helmet"]], mode=2, required_ppe=["helmet"]
        )
        self.assertIn("VIOLATION", status)
        self.assertIn("helmet", status)

    def test_sounds_play_only_when_overall_status_changes(self):
        app = SimpleNamespace(last_audio_status=None, audio_enabled=True)
        with patch.object(ppe.mixer.music, "load") as load, \
             patch.object(ppe.mixer.music, "play") as play:
            for status in ("PASS", "PASS", "VIOLATION", "VIOLATION", None, "PASS"):
                ppe.PPEDetectionApp.play_status_sound(app, status)
        self.assertEqual([call.args[0] for call in load.call_args_list],
                         [ppe.PASS_SOUND_PATH, ppe.FAIL_SOUND_PATH, ppe.PASS_SOUND_PATH])
        self.assertEqual(play.call_count, 3)

    def test_audio_disabled_does_not_attempt_playback(self):
        app = SimpleNamespace(last_audio_status=None, audio_enabled=False)
        with patch.object(ppe.mixer.music, "load") as load:
            ppe.PPEDetectionApp.play_status_sound(app, "PASS")
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()