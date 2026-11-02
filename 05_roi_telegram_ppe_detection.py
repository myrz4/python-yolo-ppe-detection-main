"""
YOLO PPE Detection with ROI, Alerts and Telegram Notification
Complete PPE detection system with visual/audio alerts and Telegram notifications
"""

import sys
import cv2
from datetime import datetime
from ultralytics import YOLO
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QComboBox, QTextEdit,
                               QGroupBox, QRadioButton, QLineEdit, QCheckBox)
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from pygame import mixer
import requests
import os
import numpy as np

# ==================== CONFIGURATION ====================
MODEL_PATH = 'models/best.pt'
CONFIDENCE_THRESHOLD = 0.25
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file, or RTSP URL

# Alert sound file (MP3)
ALERT_SOUND_PATH = 'alert.mp3'

# Alert cooldown (seconds)
ALERT_COOLDOWN = 3

# Telegram notification cooldown (seconds) - separate from audio alert
TELEGRAM_COOLDOWN = 30  # Send Telegram notification max once per 30 seconds

# Telegram Bot Configuration
# Get your bot token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN_HERE'
# Get your chat ID from @userinfobot on Telegram
TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID_HERE'

# Violation detection mode
VIOLATION_MODE = 1  # 1: Any missing PPE, 2: Specific required PPE

# Class names
PPE_PROPER = ['helmet', 'gloves', 'vest', 'boots', 'goggles']
PPE_VIOLATIONS = ['no_helmet', 'no_goggle', 'no_gloves', 'no_boots', 'none']
PERSON_CLASS = 'Person'

# Mode 2: Specific rules
REQUIRED_PPE = ['helmet', 'vest']
# =======================================================


class TelegramNotifier:
    """Handle Telegram notifications"""

    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.enabled = self.validate_credentials()

    def validate_credentials(self):
        """Validate Telegram bot credentials"""
        if not self.bot_token or self.bot_token == 'YOUR_BOT_TOKEN_HERE':
            print("Warning: Telegram bot token not configured")
            return False
        if not self.chat_id or self.chat_id == 'YOUR_CHAT_ID_HERE':
            print("Warning: Telegram chat ID not configured")
            return False

        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=5)
            if response.status_code == 200:
                print("Telegram bot credentials validated successfully")
                return True
            else:
                print(f"Telegram bot validation failed: {response.text}")
                return False
        except Exception as e:
            print(f"Error validating Telegram credentials: {str(e)}")
            return False

    def send_message(self, message):
        """Send text message to Telegram"""
        if not self.enabled:
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending Telegram message: {str(e)}")
            return False

    def send_photo(self, image, caption):
        """Send photo with caption to Telegram"""
        if not self.enabled:
            return False

        try:
            url = f"{self.base_url}/sendPhoto"

            # Convert numpy array to JPEG bytes
            _, img_encoded = cv2.imencode('.jpg', image)
            img_bytes = img_encoded.tobytes()

            files = {'photo': ('violation.jpg', img_bytes, 'image/jpeg')}
            data = {
                "chat_id": self.chat_id,
                "caption": caption,
                "parse_mode": "HTML"
            }

            response = requests.post(url, files=files, data=data, timeout=15)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending Telegram photo: {str(e)}")
            return False


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

        label_width = self.width()
        label_height = self.height()

        scale_x = frame_width / label_width
        scale_y = frame_height / label_height

        x1 = int(min(self.roi_start.x(), self.roi_end.x()) * scale_x)
        y1 = int(min(self.roi_start.y(), self.roi_end.y()) * scale_y)
        x2 = int(max(self.roi_start.x(), self.roi_end.x()) * scale_x)
        y2 = int(max(self.roi_start.y(), self.roi_end.y()) * scale_y)

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
        self.setWindowTitle("PPE Detection with ROI, Alerts and Telegram")
        self.setGeometry(100, 100, 1500, 950)

        # Initialize variables
        self.model = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.use_roi = False
        self.last_alert_time = None
        self.last_telegram_time = None
        self.violation_mode = VIOLATION_MODE
        self.current_frame = None

        # Initialize Telegram
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

        # Initialize audio mixer
        try:
            mixer.init()
            self.audio_enabled = os.path.exists(ALERT_SOUND_PATH)
            if not self.audio_enabled:
                print(f"Warning: Alert sound file not found at {ALERT_SOUND_PATH}")
        except Exception as e:
            print(f"Warning: Could not initialize audio mixer - {str(e)}")
            self.audio_enabled = False

        # Setup UI
        self.setup_ui()

        # Load model
        self.load_model()

    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Video display
        left_layout = QVBoxLayout()

        self.video_label = VideoLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid black;")
        left_layout.addWidget(self.video_label)

        # Controls
        control_layout = QHBoxLayout()

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam", "RTSP Stream", "Video File"])
        control_layout.addWidget(QLabel("Source:"))
        control_layout.addWidget(self.source_combo)

        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.toggle_detection)
        control_layout.addWidget(self.start_btn)

        self.roi_btn = QPushButton("Draw ROI")
        self.roi_btn.clicked.connect(self.toggle_roi_mode)
        self.roi_btn.setEnabled(False)
        control_layout.addWidget(self.roi_btn)

        self.clear_roi_btn = QPushButton("Clear ROI")
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        self.clear_roi_btn.setEnabled(False)
        control_layout.addWidget(self.clear_roi_btn)

        control_layout.addStretch()
        left_layout.addLayout(control_layout)

        self.status_label = QLabel("Status: Ready")
        left_layout.addWidget(self.status_label)

        main_layout.addLayout(left_layout, 3)

        # Right panel - Settings and alerts
        right_layout = QVBoxLayout()

        # Telegram settings
        telegram_group = QGroupBox("Telegram Settings")
        telegram_layout = QVBoxLayout()

        telegram_status = "Connected" if self.telegram.enabled else "Not configured"
        status_color = "green" if self.telegram.enabled else "red"
        self.telegram_status_label = QLabel(f"Status: <b style='color:{status_color}'>{telegram_status}</b>")
        telegram_layout.addWidget(self.telegram_status_label)

        self.telegram_enabled_checkbox = QCheckBox("Enable Telegram Notifications")
        self.telegram_enabled_checkbox.setChecked(self.telegram.enabled)
        self.telegram_enabled_checkbox.setEnabled(self.telegram.enabled)
        telegram_layout.addWidget(self.telegram_enabled_checkbox)

        telegram_info = QLabel("Note: Configure bot token and chat ID in the script")
        telegram_info.setStyleSheet("color: gray; font-size: 10px;")
        telegram_layout.addWidget(telegram_info)

        telegram_group.setLayout(telegram_layout)
        right_layout.addWidget(telegram_group)

        # Violation mode settings
        mode_group = QGroupBox("Violation Detection Mode")
        mode_layout = QVBoxLayout()

        self.mode1_radio = QRadioButton("Mode 1: Alert on ANY missing PPE")
        self.mode2_radio = QRadioButton("Mode 2: Alert on specific required PPE only")
        self.mode1_radio.setChecked(self.violation_mode == 1)
        self.mode2_radio.setChecked(self.violation_mode == 2)
        self.mode1_radio.toggled.connect(self.on_mode_changed)

        mode_layout.addWidget(self.mode1_radio)
        mode_layout.addWidget(self.mode2_radio)

        required_label = QLabel(f"Mode 2 Required PPE: {', '.join(REQUIRED_PPE)}")
        required_label.setStyleSheet("color: gray; font-size: 10px;")
        mode_layout.addWidget(required_label)

        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)

        # Violation alert log
        alert_group = QGroupBox("Violation Alerts")
        alert_layout = QVBoxLayout()

        self.alert_log = QTextEdit()
        self.alert_log.setReadOnly(True)
        self.alert_log.setMaximumHeight(250)
        alert_layout.addWidget(self.alert_log)

        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_alert_log)
        alert_layout.addWidget(self.clear_log_btn)

        alert_group.setLayout(alert_layout)
        right_layout.addWidget(alert_group)

        # Current detections
        detection_group = QGroupBox("Current Detections")
        detection_layout = QVBoxLayout()

        self.detection_label = QLabel("No detections")
        self.detection_label.setWordWrap(True)
        detection_layout.addWidget(self.detection_label)

        detection_group.setLayout(detection_layout)
        right_layout.addWidget(detection_group)

        right_layout.addStretch()
        main_layout.addLayout(right_layout, 1)

    def load_model(self):
        """Load YOLO model"""
        try:
            self.status_label.setText(f"Status: Loading model from {MODEL_PATH}...")
            self.model = YOLO(MODEL_PATH)
            self.status_label.setText("Status: Model loaded successfully")
        except Exception as e:
            self.status_label.setText(f"Status: Error loading model - {str(e)}")

    def on_mode_changed(self):
        """Handle violation mode change"""
        if self.mode1_radio.isChecked():
            self.violation_mode = 1
        else:
            self.violation_mode = 2

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
        self.timer.start(30)

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

    def clear_alert_log(self):
        """Clear the alert log"""
        self.alert_log.clear()

    def check_violations(self, results):
        """
        Check for PPE violations based on detection results
        Returns: (has_violation, violation_details)
        """
        boxes = results[0].boxes
        detected_classes = []

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            detected_classes.append(class_name)

        # Mode 1: Any missing PPE or violation class detected
        if self.violation_mode == 1:
            for violation in PPE_VIOLATIONS:
                if violation in detected_classes:
                    return True, f"Violation detected: {violation}"

            if PERSON_CLASS in detected_classes:
                missing_ppe = [ppe for ppe in PPE_PROPER if ppe not in detected_classes]
                if missing_ppe:
                    return True, f"Person detected without: {', '.join(missing_ppe)}"

        # Mode 2: Check only for specific required PPE
        else:
            violation_map = {
                'helmet': 'no_helmet',
                'goggles': 'no_goggle',
                'gloves': 'no_gloves',
                'boots': 'no_boots'
            }

            for required in REQUIRED_PPE:
                violation_class = violation_map.get(required)
                if violation_class and violation_class in detected_classes:
                    return True, f"Required PPE violation: {violation_class}"

            if PERSON_CLASS in detected_classes:
                missing_required = [ppe for ppe in REQUIRED_PPE if ppe not in detected_classes]
                if missing_required:
                    return True, f"Person missing required PPE: {', '.join(missing_required)}"

        return False, None

    def send_telegram_notification(self, violation_message):
        """Send Telegram notification with photo"""
        if not self.telegram_enabled_checkbox.isChecked():
            return

        current_time = datetime.now()

        # Check Telegram cooldown
        if self.last_telegram_time:
            time_diff = (current_time - self.last_telegram_time).total_seconds()
            if time_diff < TELEGRAM_COOLDOWN:
                return

        self.last_telegram_time = current_time

        # Prepare message
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

        message = f"""
<b>⚠️ PPE VIOLATION ALERT</b>

<b>Time:</b> {timestamp}
<b>Location:</b> Monitoring Area
<b>Violation:</b> {violation_message}

<b>Action Required:</b>
Please ensure all personnel wear proper PPE equipment in the designated area.

<i>This is an automated safety notification from your PPE Detection System.</i>
"""

        # Send photo with caption
        if self.current_frame is not None:
            success = self.telegram.send_photo(self.current_frame, message)
            if success:
                self.alert_log.append(f"[{timestamp}] Telegram notification sent successfully")
            else:
                self.alert_log.append(f"[{timestamp}] Failed to send Telegram notification")

    def trigger_alert(self, violation_message):
        """Trigger audio and text alert"""
        current_time = datetime.now()

        # Check alert cooldown
        if self.last_alert_time:
            time_diff = (current_time - self.last_alert_time).total_seconds()
            if time_diff < ALERT_COOLDOWN:
                return

        self.last_alert_time = current_time

        # Add to log
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] ALERT: {violation_message}"
        self.alert_log.append(log_message)

        # Play audio alert
        if self.audio_enabled:
            try:
                mixer.music.load(ALERT_SOUND_PATH)
                mixer.music.play()
            except Exception as e:
                print(f"Error playing alert sound: {str(e)}")

        # Send Telegram notification
        self.send_telegram_notification(violation_message)

        # Update status
        self.status_label.setText(f"Status: VIOLATION DETECTED - {violation_message}")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")

    def update_frame(self):
        """Update video frame with detection"""
        ret, frame = self.cap.read()

        if not ret:
            self.stop_detection()
            return

        frame_height, frame_width = frame.shape[:2]

        # Apply ROI if set
        detection_frame = frame.copy()
        roi_coords = None

        if self.use_roi:
            roi_coords = self.video_label.get_roi_coordinates(frame_width, frame_height)
            if roi_coords:
                x1, y1, x2, y2 = roi_coords
                detection_frame = frame[y1:y2, x1:x2]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "ROI", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Run detection
        if detection_frame.size > 0:
            results = self.model.predict(source=detection_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # Check for violations
            has_violation, violation_message = self.check_violations(results)

            # Get annotated frame (with bounding boxes) BEFORE triggering alert
            if self.use_roi and roi_coords:
                x1, y1, x2, y2 = roi_coords
                annotated_roi = results[0].plot()
                frame[y1:y2, x1:x2] = annotated_roi
                annotated_frame = frame
            else:
                annotated_frame = results[0].plot()

            # Update stored frame with annotations (for Telegram with bounding boxes)
            self.current_frame = annotated_frame.copy()

            if has_violation:
                self.trigger_alert(violation_message)
            else:
                self.status_label.setText("Status: Detection running... No violations")
                self.status_label.setStyleSheet("")

            # Update detection display
            boxes = results[0].boxes
            detected_items = []
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                detected_items.append(f"{class_name} ({confidence:.2f})")

            if detected_items:
                self.detection_label.setText("Detected:\n" + "\n".join(detected_items))
            else:
                self.detection_label.setText("No detections")

            # Add violation warning overlay if violation detected
            if has_violation:
                cv2.putText(annotated_frame, "VIOLATION DETECTED!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            annotated_frame = frame

        # Convert frame to QPixmap and display
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_detection()
        if self.audio_enabled:
            mixer.quit()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = PPEDetectionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
