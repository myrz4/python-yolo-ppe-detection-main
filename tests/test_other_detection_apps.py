import importlib.util
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
from PySide6.QtWidgets import QApplication

import ppe_status


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "04_roi_alert_ppe_detection.py",
    "05_roi_telegram_ppe_detection.py",
    "06_roi_email_logging_ppe_detection.py",
    "06_roi_telegram_logging_ppe_detection.py",
)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name[:-3], ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Box:
    def __init__(self, class_id, bbox):
        self.cls = [class_id]
        self.conf = [0.9]
        self.xyxy = [SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(bbox)))]


class OtherDetectionAppsTests(unittest.TestCase):
    def run_frame(self, script, people, mode=1, use_roi=False, required_ppe=None):
        script_module = load_script(script)
        names = [ppe_status.PERSON_CLASS, *ppe_status.PPE_PROPER, *ppe_status.PPE_VIOLATIONS]
        boxes = []
        for i, items in enumerate(people):
            x = i * 200
            boxes.append(Box(0, [x, 0, x + 150, 200]))
            boxes.extend(Box(names.index(item), [x + 20, 20, x + 40, 40]) for item in items)
        frame = np.zeros((300, 700, 3), dtype=np.uint8)
        roi = (30, 40, 670, 280)
        result = SimpleNamespace(boxes=boxes, plot=lambda: np.zeros(
            (240, 640, 3) if use_roi else frame.shape, dtype=np.uint8
        ))
        video = Mock()
        video.roi_active = use_roi
        video.roi_start = (30, 40) if use_roi else None
        video.roi_end = (670, 280) if use_roi else None
        video.drawing = True
        video.get_roi_coordinates.return_value = roi
        is_email = script.startswith("06_")
        person_ids = list(range(19, 19 + len(people)))
        app = SimpleNamespace(
            cap=SimpleNamespace(read=lambda: (True, frame.copy())),
            model=SimpleNamespace(names=names, predict=lambda **kwargs: [result]),
            frame_count=0,
            use_roi=use_roi,
            violation_mode=mode,
            required_ppe=list(
                script_module.REQUIRED_PPE if required_ppe is None else required_ppe
            ),
            video_label=video,
            status_label=Mock(),
            detection_label=Mock(),
            play_status_sound=Mock(),
            trigger_alert=Mock(),
            person_tracker=(
                SimpleNamespace(update=lambda bboxes: person_ids)
                if is_email else ppe_status.PersonTracker()
            ),
            logged_person_ids=set(person_ids) if is_email else set(),
        )
        if hasattr(script_module.PPEDetectionApp, "_check_proximity"):
            app._check_proximity = lambda a, b: (
                script_module.PPEDetectionApp._check_proximity(app, a, b)
            )
        app.check_violations = lambda results: script_module.PPEDetectionApp.check_violations(app, results)
        with patch.object(script_module.cv2, "putText") as put_text, \
             patch.object(script_module, "QImage"), patch.object(script_module, "QPixmap"):
            script_module.PPEDetectionApp.update_frame(app)
        return app, put_text.call_args_list

    def test_statuses_labels_and_sound_for_all_scripts(self):
        for script in SCRIPTS:
            for people, expected, labels in (
                ([ppe_status.PPE_PROPER], "PASS", ["PASS"]),
                ([["helmet"]], "VIOLATION", ["VIOLATION"]),
                ([ppe_status.PPE_PROPER, ["helmet"]], "VIOLATION", ["PASS", "VIOLATION"]),
                ([], None, []),
            ):
                with self.subTest(script=script, people=people):
                    app, draws = self.run_frame(script, people)
                    app.play_status_sound.assert_called_once_with(expected)
                    texts = [draw.args[1] for draw in draws]
                    id_base = 19 if script.startswith("06_") else 1
                    expected_people = [f"ID:{id_base + i}" for i in range(len(people))]
                    self.assertEqual(texts, expected_people + ([expected] if expected else []))
                    if expected is None:
                        self.assertIn("No person detected", app.status_label.setText.call_args.args[0])
                    else:
                        self.assertIn(expected, app.status_label.setText.call_args.args[0])
                    self.assertEqual(app.trigger_alert.call_count, int(expected == "VIOLATION"))

    def test_required_ppe_and_explicit_violation(self):
        for script in SCRIPTS:
            for items, mode, expected, required_ppe in (
                (["helmet", "vest"], 2, "PASS", None),
                (["helmet", "vest", "no_helmet"], 2, "VIOLATION", None),
                (ppe_status.PPE_PROPER + ["no_gloves"], 1, "VIOLATION", None),
                (["helmet", "vest", "no_boots"], 2, "PASS", None),
                (["helmet", "vest"], 2, "PASS", ["helmet"]),
                (["vest", "no_helmet"], 2, "VIOLATION", ["helmet"]),
                (["helmet", "no_boots"], 2, "PASS", ["helmet"]),
            ):
                with self.subTest(script=script, items=items):
                    app, _ = self.run_frame(
                        script, [items], mode, required_ppe=required_ppe
                    )
                    app.play_status_sound.assert_called_once_with(expected)

    def test_ppe_is_not_shared_between_people_and_orphan_detections_are_ignored(self):
        names = [ppe_status.PERSON_CLASS, *ppe_status.PPE_PROPER, *ppe_status.PPE_VIOLATIONS]
        boxes = [Box(0, [0, 0, 150, 200]), Box(0, [200, 0, 350, 200])]
        boxes.extend(Box(names.index(item), [20, 20, 40, 40]) for item in ppe_status.PPE_PROPER)
        boxes.append(Box(names.index("no_helmet"), [500, 20, 520, 40]))
        results = [SimpleNamespace(boxes=boxes)]
        people = ppe_status.check_violations(results, names, 1, ["helmet", "vest"])
        self.assertFalse(people[0]['has_violation'])
        self.assertTrue(people[1]['has_violation'])
        self.assertIn("helmet", people[1]['violation_message'])
        self.assertEqual(ppe_status.check_violations(
            [SimpleNamespace(boxes=[boxes[-1]])], names, 1, ["helmet", "vest"]
        ), [])

    def test_roi_offsets_person_labels(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                _, draws = self.run_frame(script, [ppe_status.PPE_PROPER], use_roi=True)
                person_label = "ID:19" if script.startswith("06_") else "ID:1"
                draw = next(draw for draw in draws if draw.args[1] == person_label and draw.args[3] == 0)
                self.assertEqual(draw.args[2], (30, 30))

    def test_status_audio_does_not_repeat_and_resets_after_no_person(self):
        app = SimpleNamespace(last_audio_status=None, audio_enabled=True)
        with patch.object(ppe_status.mixer.music, "load") as load, \
             patch.object(ppe_status.mixer.music, "play") as play:
            for status in ("PASS", "PASS", "VIOLATION", "VIOLATION", None, "PASS"):
                ppe_status.StatusAudioMixin.play_status_sound(app, status)
        self.assertEqual(load.call_args_list, [
            call(ppe_status.PASS_SOUND_PATH),
            call(ppe_status.FAIL_SOUND_PATH),
            call(ppe_status.PASS_SOUND_PATH),
        ])
        self.assertEqual(play.call_count, 3)

    def test_tracker_keeps_ids_through_short_gaps_and_expires_old_ids(self):
        tracker = ppe_status.PersonTracker(max_distance=50, memory_frames=2)
        self.assertEqual(tracker.update([(0, 0, 20, 20)]), [1])
        self.assertEqual(tracker.update([(10, 0, 20, 20)]), [1])
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.update([(20, 0, 20, 20)]), [1])
        tracker.update([])
        tracker.update([])
        tracker.update([])
        self.assertEqual(tracker.update([(20, 0, 20, 20)]), [2])

    def test_windows_initialize_with_shared_audio_mixin(self):
        qt_app = QApplication.instance() or QApplication([])
        for script in SCRIPTS:
            with self.subTest(script=script), ExitStack() as stack:
                module = load_script(script)
                stack.enter_context(patch.object(module.PPEDetectionApp, "load_model"))
                stack.enter_context(patch.object(ppe_status.mixer, "init"))
                if script.startswith("05_"):
                    notifier = stack.enter_context(patch.object(module, "TelegramNotifier"))
                    notifier.return_value.enabled = False
                if script.startswith("06_roi_email"):
                    stack.enter_context(patch.object(module, "Database"))
                    notifier = stack.enter_context(patch.object(module, "EmailNotifier"))
                    notifier.return_value.enabled = False
                elif script.startswith("06_"):
                    stack.enter_context(patch.object(module, "Database"))
                    notifier = stack.enter_context(patch.object(module, "TelegramNotifier"))
                    notifier.return_value.enabled = False
                window = module.PPEDetectionApp()
                self.assertTrue(window.audio_enabled)
                self.assertIsNone(window.last_audio_status)
                logo = window.findChild(module.QLabel, "technipfmcLogo")
                if script.startswith(("04_", "05_", "06_")) and script != "06_roi_telegram_logging_ppe_detection.py":
                    self.assertIsNotNone(logo)
                    self.assertFalse(logo.pixmap().isNull())
                window.deleteLater()
        qt_app.processEvents()


if __name__ == "__main__":
    unittest.main()
