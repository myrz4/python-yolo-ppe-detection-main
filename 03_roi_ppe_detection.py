"""
YOLO PPE Detection with ROI using PySide6
Interactive GUI for PPE detection with Region of Interest (ROI) selection
"""

import sys
import cv2
from ultralytics import YOLO
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QComboBox)
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen

# ==================== CONFIGURATION ====================
MODEL_PATH = 'models/best.pt'
CONFIDENCE_THRESHOLD = 0.25
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file, or RTSP URL

# Class names
PPE_CLASSES = {
    'proper': ['helmet', 'gloves', 'vest', 'boots', 'goggles', 'Person'],
    'violations': ['no_helmet', 'no_goggle', 'no_gloves', 'no_boots', 'none']
}
# =======================================================


class VideoLabel(QLabel):
    """Custom QLabel for displaying video and drawing ROI"""

    def __init__(self):
        super().__init__()
        self.roi_start = None
        self.roi_end = None
        self.roi_active = False
        self.drawing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.roi_active:
            self.drawing = True
            self.roi_start = event.position().toPoint()
            self.roi_end = None

    def mouseMoveEvent(self, event):
        if self.drawing and self.roi_active:
            self.roi_end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.roi_end = event.position().toPoint()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        # Draw ROI rectangle if being drawn or already set
        if self.roi_active and self.roi_start and self.roi_end:
            painter = QPainter(self)
            pen = QPen(Qt.green, 2, Qt.SolidLine)
            painter.setPen(pen)

            x1 = min(self.roi_start.x(), self.roi_end.x())
            y1 = min(self.roi_start.y(), self.roi_end.y())
            x2 = max(self.roi_start.x(), self.roi_end.x())
            y2 = max(self.roi_start.y(), self.roi_end.y())

            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    def get_roi_coordinates(self, frame_width, frame_height):
        """Convert ROI coordinates from label to frame coordinates"""
        if not (self.roi_start and self.roi_end):
            return None

        # Get label size
        label_width = self.width()
        label_height = self.height()

        # Calculate scale factors
        scale_x = frame_width / label_width
        scale_y = frame_height / label_height

        # Get ROI coordinates
        x1 = int(min(self.roi_start.x(), self.roi_end.x()) * scale_x)
        y1 = int(min(self.roi_start.y(), self.roi_end.y()) * scale_y)
        x2 = int(max(self.roi_start.x(), self.roi_end.x()) * scale_x)
        y2 = int(max(self.roi_start.y(), self.roi_end.y()) * scale_y)

        # Ensure within frame bounds
        x1 = max(0, min(x1, frame_width))
        y1 = max(0, min(y1, frame_height))
        x2 = max(0, min(x2, frame_width))
        y2 = max(0, min(y2, frame_height))

        return (x1, y1, x2, y2)

    def clear_roi(self):
        """Clear the ROI"""
        self.roi_start = None
        self.roi_end = None
        self.update()


class PPEDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPE Detection with ROI")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.model = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.use_roi = False

        # Setup UI
        self.setup_ui()

        # Load model
        self.load_model()

    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Video display
        self.video_label = VideoLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid black;")
        main_layout.addWidget(self.video_label)

        # Controls
        control_layout = QHBoxLayout()

        # Video source selector
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam", "RTSP Stream", "Video File"])
        control_layout.addWidget(QLabel("Source:"))
        control_layout.addWidget(self.source_combo)

        # Start/Stop button
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.toggle_detection)
        control_layout.addWidget(self.start_btn)

        # ROI button
        self.roi_btn = QPushButton("Draw ROI")
        self.roi_btn.clicked.connect(self.toggle_roi_mode)
        self.roi_btn.setEnabled(False)
        control_layout.addWidget(self.roi_btn)

        # Clear ROI button
        self.clear_roi_btn = QPushButton("Clear ROI")
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        self.clear_roi_btn.setEnabled(False)
        control_layout.addWidget(self.clear_roi_btn)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # Status label
        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)

    def load_model(self):
        """Load YOLO model"""
        try:
            self.status_label.setText(f"Status: Loading model from {MODEL_PATH}...")
            self.model = YOLO(MODEL_PATH)
            self.status_label.setText("Status: Model loaded successfully")
        except Exception as e:
            self.status_label.setText(f"Status: Error loading model - {str(e)}")

    def toggle_detection(self):
        """Start or stop detection"""
        if self.cap is None or not self.cap.isOpened():
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        """Start video capture and detection"""
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)

        if not self.cap.isOpened():
            self.status_label.setText("Status: Error - Cannot open video source")
            return

        self.start_btn.setText("Stop Detection")
        self.roi_btn.setEnabled(True)
        self.clear_roi_btn.setEnabled(True)
        self.status_label.setText("Status: Detection running...")
        self.timer.start(30)  # Update every 30ms (~33 FPS)

    def stop_detection(self):
        """Stop video capture and detection"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.cap = None
        self.start_btn.setText("Start Detection")
        self.roi_btn.setEnabled(False)
        self.clear_roi_btn.setEnabled(False)
        self.video_label.clear()
        self.status_label.setText("Status: Detection stopped")

    def toggle_roi_mode(self):
        """Toggle ROI drawing mode"""
        if not self.video_label.roi_active:
            self.video_label.roi_active = True
            self.roi_btn.setText("Disable ROI Mode")
            self.use_roi = True
            self.status_label.setText("Status: Draw ROI by clicking and dragging on the video")
        else:
            self.video_label.roi_active = False
            self.roi_btn.setText("Draw ROI")
            self.status_label.setText("Status: ROI mode disabled")

    def clear_roi(self):
        """Clear the ROI"""
        self.video_label.clear_roi()
        self.use_roi = False
        self.status_label.setText("Status: ROI cleared")

    def update_frame(self):
        """Update video frame with detection"""
        ret, frame = self.cap.read()

        if not ret:
            self.stop_detection()
            return

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Apply ROI if set
        detection_frame = frame.copy()
        roi_coords = None

        if self.use_roi:
            roi_coords = self.video_label.get_roi_coordinates(frame_width, frame_height)
            if roi_coords:
                x1, y1, x2, y2 = roi_coords
                detection_frame = frame[y1:y2, x1:x2]

                # Draw ROI rectangle on original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "ROI", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Run detection
        if detection_frame.size > 0:
            results = self.model.predict(source=detection_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # Get annotated frame
            if self.use_roi and roi_coords:
                x1, y1, x2, y2 = roi_coords
                annotated_roi = results[0].plot()
                # Place annotated ROI back into original frame
                frame[y1:y2, x1:x2] = annotated_roi
                annotated_frame = frame
            else:
                annotated_frame = results[0].plot()
        else:
            annotated_frame = frame

        # Convert frame to QPixmap and display
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_detection()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = PPEDetectionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
