import importlib.util
import os
from pathlib import Path
import tempfile
import tkinter
from tkinter import filedialog
import unittest
from unittest.mock import Mock, patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "08_capture_dataset_images.py"
spec = importlib.util.spec_from_file_location("capture_dataset_images", SCRIPT)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class DatasetCaptureTests(unittest.TestCase):
    def test_default_and_explicit_conditions(self):
        self.assertEqual(capture.choose_condition(None), "full_ppe")
        self.assertEqual(capture.choose_condition("2"), "no_helmet")
        self.assertEqual(capture.choose_condition("custom"), "custom")

    def test_folder_picker_starts_at_current_folder_and_cleans_up(self):
        root = Mock()
        with patch.object(tkinter, "Tk", return_value=root), \
             patch.object(filedialog, "askdirectory", return_value="chosen") as dialog:
            self.assertEqual(capture.choose_folder("."), "chosen")
        dialog.assert_called_once_with(
            parent=root, title="Select or create a parent dataset folder",
            initialdir=os.path.abspath("."), mustexist=True,
        )
        root.withdraw.assert_called_once()
        root.destroy.assert_called_once()

    def run_capture(self, output, keys, selected):
        frame = np.zeros((120, 640, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)
        argv = [
            str(SCRIPT), "--output", output, "--interval", "0", "--start-delay", "0",
            "--min-sharpness", "0",
        ]
        with patch("sys.argv", argv), \
             patch.object(capture, "open_source", return_value=camera), \
             patch.object(capture, "choose_folder",
                          side_effect=selected if isinstance(selected, list) else None,
                          return_value=selected) as picker, \
             patch.object(capture.cv2, "imshow"), \
             patch.object(capture.cv2, "waitKey", side_effect=[ord(key) for key in keys]), \
             patch.object(capture.cv2, "destroyAllWindows"), \
             patch.object(capture.cv2, "imwrite", return_value=True) as write:
            capture.main()
        camera.release.assert_called_once()
        return picker, [call.args[0] for call in write.call_args_list]

    def test_selected_parent_contains_current_and_switched_modes(self):
        with tempfile.TemporaryDirectory() as output:
            selected = os.path.join(output, "person02")
            os.mkdir(selected)
            existing = os.path.join(selected, "no_helmet")
            os.mkdir(existing)
            marker = os.path.join(existing, "existing.txt")
            Path(marker).touch()
            picker, paths = self.run_capture(output, "f21q", selected)
            picker.assert_called_once_with(os.path.join(output, "person01"))
            self.assertEqual([os.path.dirname(path) for path in paths], [
                os.path.join(output, "person01", "full_ppe"),
                os.path.join(selected, "full_ppe"),
                existing,
                os.path.join(selected, "full_ppe"),
            ])
            self.assertTrue(os.path.isdir(os.path.join(selected, "full_ppe")))
            self.assertTrue(os.path.isfile(marker))
            self.assertTrue(os.path.basename(paths[1]).startswith("person02_full_ppe_"))
            self.assertTrue(os.path.basename(paths[2]).startswith("person02_no_helmet_"))

    def test_cancel_keeps_folder_and_duplicate_filter(self):
        with tempfile.TemporaryDirectory() as output:
            _, paths = self.run_capture(output, "fsq", "")
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(os.path.dirname(path) == os.path.join(
                output, "person01", "full_ppe") for path in paths))

    def test_cancel_after_selection_keeps_selected_parent(self):
        with tempfile.TemporaryDirectory() as output:
            selected = os.path.join(output, "person02")
            os.mkdir(selected)
            picker, paths = self.run_capture(output, "ff2q", [selected, ""])
            self.assertEqual(picker.call_args_list[1].args[0], selected)
            self.assertEqual(os.path.dirname(paths[-1]), os.path.join(selected, "no_helmet"))


if __name__ == "__main__":
    unittest.main()
