"""
YOLO PPE Detection with ROI, Alerts, Email and SQLite Logging
Complete PPE detection system with database logging, person tracking, and reporting
"""

import sys
import cv2
import sqlite3
import os
from datetime import datetime, date
from ultralytics import YOLO
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QComboBox, QTextEdit,
                               QGroupBox, QRadioButton, QCheckBox, QLineEdit)
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from pygame import mixer
import pytz
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# ==================== CONFIGURATION ====================
# Timezone Configuration
TIMEZONE = pytz.timezone('Asia/Kuala_Lumpur')

def get_local_now():
    """Get current time in Asia/Kuala_Lumpur timezone"""
    return datetime.now(TIMEZONE)

def get_local_date():
    """Get current date in Asia/Kuala_Lumpur timezone"""
    return get_local_now().date()

MODEL_PATH = 'models/best.pt'
CONFIDENCE_THRESHOLD = 0.25
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file, or RTSP URL

# Camera/Zone Configuration
CAMERA_ZONE_NAME = 'Main Entrance'  # Change this for different monitoring zones

# Alert sound file (MP3)
ALERT_SOUND_PATH = 'alert.mp3'

# Alert cooldown (seconds)
ALERT_COOLDOWN = 3

# Email notification cooldown (seconds)
EMAIL_COOLDOWN = 30

# Performance Configuration
MAX_FRAME_WIDTH = 1280  # Resize frames larger than this for faster processing

# Email Configuration
EMAIL_SENDER = 'YOUR_EMAIL_SENDER'  # Sender email address
EMAIL_PASSWORD = 'YOUR_EMAIL_PASSWORD'  # App password (not regular password)
EMAIL_RECIPIENT = 'YOUR_EMAIL_RECIPIENT'  # Recipient email address
SMTP_SERVER = 'YOUR_EMAIL_HOST'  # SMTP server
SMTP_PORT = 587  # SMTP port (587 for TLS)

# Violation detection mode
VIOLATION_MODE = 1  # 1: Any missing PPE, 2: Specific required PPE

# Database configuration
DATABASE_PATH = 'data/ppe_detection.db'
IMAGES_BASE_PATH = 'data/violations'

# Person tracking configuration
PERSON_TRACKING_ENABLED = True
# Distance threshold for matching persons (in pixels)
TRACKING_DISTANCE_THRESHOLD = 50
# Frames to keep person in memory after disappearing from ROI
TRACKING_MEMORY_FRAMES = 30

# Class names
PPE_PROPER = ['helmet', 'gloves', 'vest', 'boots', 'goggles']
PPE_VIOLATIONS = ['no_helmet', 'no_goggle', 'no_gloves', 'no_boots', 'none']
PERSON_CLASS = 'Person'

# Mode 2: Specific rules
REQUIRED_PPE = ['helmet', 'vest']
# =======================================================


class Database:
    """Handle SQLite database operations"""

    def __init__(self, db_path):
        self.db_path = db_path
        self.ensure_database()

    def ensure_database(self):
        """Create database and tables if they don't exist"""
        # Create data directory if not exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                zone_name TEXT,
                violation_type TEXT,
                person_id INTEGER,
                confidence REAL,
                image_path TEXT
            )
        ''')

        # Detections table - tracks all people detected
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                zone_name TEXT,
                person_id INTEGER,
                has_helmet BOOLEAN,
                has_vest BOOLEAN,
                has_gloves BOOLEAN,
                has_boots BOOLEAN,
                has_goggles BOOLEAN,
                is_compliant BOOLEAN
            )
        ''')

        # Daily statistics summary
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                zone_name TEXT,
                total_people INTEGER,
                compliant_people INTEGER,
                violation_people INTEGER,
                compliance_rate REAL,
                UNIQUE(date, zone_name)
            )
        ''')

        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")

    def log_violation(self, zone_name, violation_type, person_id, confidence, image_path):
        """Log a violation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Use local timezone for timestamp
        local_time = get_local_now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute('''
            INSERT INTO violations (timestamp, zone_name, violation_type, person_id, confidence, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (local_time, zone_name, violation_type, person_id, confidence, image_path))

        conn.commit()
        conn.close()

    def log_detection(self, zone_name, person_id, ppe_status, is_compliant):
        """Log a person detection with PPE status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Use local timezone for timestamp
        local_time = get_local_now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute('''
            INSERT INTO detections (timestamp, zone_name, person_id, has_helmet, has_vest, has_gloves,
                                   has_boots, has_goggles, is_compliant)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (local_time, zone_name, person_id, ppe_status.get('helmet', False),
              ppe_status.get('vest', False), ppe_status.get('gloves', False),
              ppe_status.get('boots', False), ppe_status.get('goggles', False),
              is_compliant))

        conn.commit()
        conn.close()

    def update_daily_stats(self, zone_name, total_people, compliant_people, violation_people):
        """Update daily statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Use local date
        today = get_local_date().isoformat()
        compliance_rate = (compliant_people / total_people * 100) if total_people > 0 else 0

        cursor.execute('''
            INSERT OR REPLACE INTO daily_stats
            (date, zone_name, total_people, compliant_people, violation_people, compliance_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (today, zone_name, total_people, compliant_people, violation_people, compliance_rate))

        conn.commit()
        conn.close()

    def get_today_stats(self, zone_name):
        """Get today's statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Use local date
        today = get_local_date().isoformat()

        cursor.execute('''
            SELECT total_people, compliant_people, violation_people, compliance_rate
            FROM daily_stats
            WHERE date = ? AND zone_name = ?
        ''', (today, zone_name))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'total_people': result[0],
                'compliant': result[1],
                'violations': result[2],
                'compliance_rate': result[3]
            }
        return {'total_people': 0, 'compliant': 0, 'violations': 0, 'compliance_rate': 0}


class PersonTracker:
    """Simple centroid-based person tracker for counting unique individuals in ROI"""

    def __init__(self, max_distance=50, memory_frames=30):
        self.next_id = 1
        self.tracked_persons = {}  # {person_id: {'centroid': (x, y), 'frames_missing': 0, 'logged': False}}
        self.max_distance = max_distance
        self.memory_frames = memory_frames

    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of (x, y, w, h) bounding boxes
        Returns: list of person_ids for each detection
        """
        # Calculate centroids from detections
        current_centroids = []
        for (x, y, w, h) in detections:
            cx = x + w // 2
            cy = y + h // 2
            current_centroids.append((cx, cy))

        # If no current detections, increment frames_missing for all tracked persons
        if len(current_centroids) == 0:
            for person_id in list(self.tracked_persons.keys()):
                self.tracked_persons[person_id]['frames_missing'] += 1
                # Remove if missing for too long
                if self.tracked_persons[person_id]['frames_missing'] > self.memory_frames:
                    del self.tracked_persons[person_id]
            return []

        # If no tracked persons yet, register all as new
        if len(self.tracked_persons) == 0:
            person_ids = []
            for centroid in current_centroids:
                person_ids.append(self._register(centroid))
            return person_ids

        # Match current centroids with tracked persons
        tracked_ids = list(self.tracked_persons.keys())
        tracked_centroids = [self.tracked_persons[pid]['centroid'] for pid in tracked_ids]

        # Calculate distance matrix
        matched_ids = []
        used_tracked_ids = set()

        for current_centroid in current_centroids:
            min_distance = float('inf')
            best_match_id = None

            for i, tracked_id in enumerate(tracked_ids):
                if tracked_id in used_tracked_ids:
                    continue

                tracked_centroid = tracked_centroids[i]
                distance = self._calculate_distance(current_centroid, tracked_centroid)

                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_match_id = tracked_id

            if best_match_id is not None:
                # Update existing person
                self.tracked_persons[best_match_id]['centroid'] = current_centroid
                self.tracked_persons[best_match_id]['frames_missing'] = 0
                matched_ids.append(best_match_id)
                used_tracked_ids.add(best_match_id)
            else:
                # Register new person
                new_id = self._register(current_centroid)
                matched_ids.append(new_id)

        # Increment frames_missing for unmatched tracked persons
        for person_id in tracked_ids:
            if person_id not in used_tracked_ids:
                self.tracked_persons[person_id]['frames_missing'] += 1
                # Remove if missing for too long
                if self.tracked_persons[person_id]['frames_missing'] > self.memory_frames:
                    del self.tracked_persons[person_id]

        return matched_ids

    def _register(self, centroid):
        """Register a new person"""
        person_id = self.next_id
        self.tracked_persons[person_id] = {
            'centroid': centroid,
            'frames_missing': 0,
            'logged': False
        }
        self.next_id += 1
        return person_id

    def _calculate_distance(self, centroid1, centroid2):
        """Calculate Euclidean distance between two centroids"""
        return ((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2) ** 0.5

    def mark_logged(self, person_id):
        """Mark person as logged to database"""
        if person_id in self.tracked_persons:
            self.tracked_persons[person_id]['logged'] = True

    def is_logged(self, person_id):
        """Check if person has been logged"""
        if person_id in self.tracked_persons:
            return self.tracked_persons[person_id]['logged']
        return False

    def get_unique_count(self):
        """Get count of unique persons currently tracked"""
        return len(self.tracked_persons)

    def reset(self):
        """Reset tracker"""
        self.tracked_persons = {}
        self.next_id = 1


class EmailNotifier:
    """Handle Email notifications with async support"""

    def __init__(self, sender_email, sender_password, recipient_email, smtp_server, smtp_port):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.enabled = self.validate_credentials()
        # Thread pool for async notifications
        self.executor = ThreadPoolExecutor(max_workers=2)

    def validate_credentials(self):
        """Validate email credentials"""
        if not self.sender_email or self.sender_email == 'YOUR_EMAIL_HERE':
            print("Warning: Email sender not configured")
            return False
        if not self.sender_password or self.sender_password == 'YOUR_EMAIL_PASSWORD_HERE':
            print("Warning: Email password not configured")
            return False
        if not self.recipient_email or self.recipient_email == 'YOUR_EMAIL_RECIPIENT':
            print("Warning: Email recipient not configured")
            return False

        try:
            # Test SMTP connection
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.quit()
            print("Email credentials validated successfully")
            return True
        except Exception as e:
            print(f"Error validating email credentials: {str(e)}")
            return False

    def send_email(self, subject, body, image=None):
        """Send email with optional image attachment"""
        if not self.enabled:
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject

            # Add body
            msg.attach(MIMEText(body, 'html'))

            # Add image if provided
            if image is not None:
                # Convert numpy array to JPEG bytes
                _, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_bytes = img_encoded.tobytes()
                
                image_attachment = MIMEImage(img_bytes, name='violation.jpg')
                msg.attach(image_attachment)

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=15)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()

            return True
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False

    def send_email_async(self, subject, body, image=None):
        """Send email asynchronously (non-blocking)"""
        if not self.enabled:
            return

        # Submit to thread pool and return immediately
        image_copy = image.copy() if image is not None else None
        self.executor.submit(self.send_email, subject, body, image_copy)


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
        self.setWindowTitle(f"PPE Detection with Logging - Zone: {CAMERA_ZONE_NAME}")
        self.setGeometry(100, 100, 1500, 950)

        # Initialize variables
        self.model = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.use_roi = False
        self.last_alert_time = None
        self.last_email_time = None
        self.violation_mode = VIOLATION_MODE
        self.current_frame = None

        # Performance tracking
        self.frame_count = 0

        # Statistics counters (for current session)
        self.session_total_people = 0
        self.session_compliant = 0
        self.session_violations = 0
        self.logged_person_ids = set()  # Track which persons we've logged to avoid duplicates

        # Initialize database
        self.db = Database(DATABASE_PATH)

        # Initialize person tracker
        self.person_tracker = PersonTracker(
            max_distance=TRACKING_DISTANCE_THRESHOLD,
            memory_frames=TRACKING_MEMORY_FRAMES
        ) if PERSON_TRACKING_ENABLED else None

        # Initialize Email
        self.email_notifier = EmailNotifier(
            EMAIL_SENDER,
            EMAIL_PASSWORD,
            EMAIL_RECIPIENT,
            SMTP_SERVER,
            SMTP_PORT
        )

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

        # Right panel - Settings, statistics and alerts
        right_layout = QVBoxLayout()

        # Zone information
        zone_group = QGroupBox("Monitoring Zone")
        zone_layout = QVBoxLayout()
        zone_layout.addWidget(QLabel(f"<b>Zone:</b> {CAMERA_ZONE_NAME}"))
        zone_layout.addWidget(QLabel(f"<b>Database:</b> {DATABASE_PATH}"))
        zone_group.setLayout(zone_layout)
        right_layout.addWidget(zone_group)

        # Session statistics
        stats_group = QGroupBox("Session Statistics")
        stats_layout = QVBoxLayout()

        self.stats_people_label = QLabel("Total People: 0")
        self.stats_compliant_label = QLabel("Compliant: 0")
        self.stats_violations_label = QLabel("Violations: 0")
        self.stats_compliance_rate_label = QLabel("Compliance Rate: 0.00%")

        stats_layout.addWidget(self.stats_people_label)
        stats_layout.addWidget(self.stats_compliant_label)
        stats_layout.addWidget(self.stats_violations_label)
        stats_layout.addWidget(self.stats_compliance_rate_label)

        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        # Email settings
        email_group = QGroupBox("Email Notification Settings")
        email_layout = QVBoxLayout()

        email_status = "Connected" if self.email_notifier.enabled else "Not configured"
        status_color = "green" if self.email_notifier.enabled else "red"
        self.email_status_label = QLabel(f"Status: <b style='color:{status_color}'>{email_status}</b>")
        email_layout.addWidget(self.email_status_label)

        email_layout.addWidget(QLabel(f"Sender: {EMAIL_SENDER}"))
        email_layout.addWidget(QLabel(f"Recipient: {EMAIL_RECIPIENT}"))

        self.email_enabled_checkbox = QCheckBox("Enable Email Notifications")
        self.email_enabled_checkbox.setChecked(self.email_notifier.enabled)
        self.email_enabled_checkbox.setEnabled(self.email_notifier.enabled)
        email_layout.addWidget(self.email_enabled_checkbox)

        email_group.setLayout(email_layout)
        right_layout.addWidget(email_group)

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
        self.alert_log.setMaximumHeight(200)
        alert_layout.addWidget(self.alert_log)

        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_alert_log)
        alert_layout.addWidget(self.clear_log_btn)

        alert_group.setLayout(alert_layout)
        right_layout.addWidget(alert_group)

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

        # Reset session statistics
        self.session_total_people = 0
        self.session_compliant = 0
        self.session_violations = 0
        self.logged_person_ids = set()
        if self.person_tracker:
            self.person_tracker.reset()
        self.update_statistics_display()

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

        # Update daily stats in database
        if self.session_total_people > 0:
            self.db.update_daily_stats(
                CAMERA_ZONE_NAME,
                self.session_total_people,
                self.session_compliant,
                self.session_violations
            )

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

    def update_statistics_display(self):
        """Update statistics display"""
        self.stats_people_label.setText(f"Total People: {self.session_total_people}")
        self.stats_compliant_label.setText(f"Compliant: {self.session_compliant}")
        self.stats_violations_label.setText(f"Violations: {self.session_violations}")

        if self.session_total_people > 0:
            compliance_rate = (self.session_compliant / self.session_total_people) * 100
            self.stats_compliance_rate_label.setText(f"Compliance Rate: {compliance_rate:.2f}%")
        else:
            self.stats_compliance_rate_label.setText("Compliance Rate: 0.00%")

    def check_violations(self, results, person_bboxes=None):
        """
        Check for PPE violations based on detection results
        Returns: list of (has_violation, violation_details, person_id, ppe_status) for each person
        """
        boxes = results[0].boxes
        detected_classes = []
        detection_data = []

        # Collect all detections
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2

            detected_classes.append(class_name)
            detection_data.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox
            })

        # Find all persons
        person_detections = [d for d in detection_data if d['class'] == PERSON_CLASS]

        results_list = []

        # Check each person for PPE compliance
        for person_det in person_detections:
            person_bbox = person_det['bbox']
            person_confidence = person_det['confidence']

            # Find PPE items near this person (simple proximity check)
            ppe_status = {
                'helmet': False,
                'vest': False,
                'gloves': False,
                'boots': False,
                'goggles': False
            }

            # Check for proper PPE near person
            for ppe_item in PPE_PROPER:
                if ppe_item == 'Person':
                    continue
                for det in detection_data:
                    if det['class'] == ppe_item:
                        # Simple check: if PPE bbox overlaps or is close to person bbox
                        if self._check_proximity(person_bbox, det['bbox']):
                            ppe_status[ppe_item] = True

            # Check for violation classes near person
            violation_detected = None
            for violation in PPE_VIOLATIONS:
                for det in detection_data:
                    if det['class'] == violation:
                        if self._check_proximity(person_bbox, det['bbox']):
                            violation_detected = violation
                            break
                if violation_detected:
                    break

            # Determine if person is compliant based on mode
            is_compliant = True
            violation_message = None

            if self.violation_mode == 1:
                # Mode 1: Any missing PPE or violation class
                if violation_detected:
                    is_compliant = False
                    violation_message = f"Violation detected: {violation_detected}"
                else:
                    missing_ppe = [ppe for ppe in PPE_PROPER if not ppe_status.get(ppe, False)]
                    if missing_ppe:
                        is_compliant = False
                        violation_message = f"Missing PPE: {', '.join(missing_ppe)}"
            else:
                # Mode 2: Check only required PPE
                if violation_detected and violation_detected in ['no_' + ppe for ppe in REQUIRED_PPE]:
                    is_compliant = False
                    violation_message = f"Required PPE violation: {violation_detected}"
                else:
                    missing_required = [ppe for ppe in REQUIRED_PPE if not ppe_status.get(ppe, False)]
                    if missing_required:
                        is_compliant = False
                        violation_message = f"Missing required PPE: {', '.join(missing_required)}"

            results_list.append({
                'has_violation': not is_compliant,
                'violation_message': violation_message,
                'ppe_status': ppe_status,
                'confidence': person_confidence,
                'bbox': person_bbox
            })

        return results_list

    def _check_proximity(self, bbox1, bbox2, threshold=0.3):
        """Check if two bounding boxes are close (simple IoU or overlap check)"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return False

        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)

        # Check if intersection is significant relative to person bbox
        overlap_ratio = inter_area / bbox1_area if bbox1_area > 0 else 0

        return overlap_ratio > threshold

    def save_violation_image(self, frame, person_id):
        """Save violation image to disk with local timezone"""
        # Create directory structure: data/violations/YYYY-MM-DD/
        today = get_local_date().isoformat()
        image_dir = os.path.join(IMAGES_BASE_PATH, today)
        os.makedirs(image_dir, exist_ok=True)

        # Generate filename with local time
        timestamp = get_local_now().strftime("%H%M%S")
        filename = f"violation_{CAMERA_ZONE_NAME.replace(' ', '_')}_{person_id}_{timestamp}.jpg"
        filepath = os.path.join(image_dir, filename)

        # Save image with compression for better performance
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return filepath

    def send_email_notification(self, violation_message):
        """Send Email notification with photo (async)"""
        if not self.email_enabled_checkbox.isChecked():
            return

        current_time = get_local_now()

        # Check Email cooldown
        if self.last_email_time:
            time_diff = (current_time - self.last_email_time).total_seconds()
            if time_diff < EMAIL_COOLDOWN:
                return

        self.last_email_time = current_time

        # Prepare email content with local timezone
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

        subject = f"⚠️ PPE VIOLATION ALERT - {CAMERA_ZONE_NAME}"

        body = f"""
<html>
<body>
    <h2 style="color: #d9534f;">⚠️ PPE VIOLATION ALERT</h2>
    
    <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
        <tr>
            <td style="padding: 8px; background-color: #f5f5f5;"><strong>Time:</strong></td>
            <td style="padding: 8px;">{timestamp} (Asia/Kuala_Lumpur)</td>
        </tr>
        <tr>
            <td style="padding: 8px; background-color: #f5f5f5;"><strong>Zone:</strong></td>
            <td style="padding: 8px;">{CAMERA_ZONE_NAME}</td>
        </tr>
        <tr>
            <td style="padding: 8px; background-color: #f5f5f5;"><strong>Violation:</strong></td>
            <td style="padding: 8px; color: #d9534f;"><strong>{violation_message}</strong></td>
        </tr>
    </table>
    
    <h3>Session Statistics:</h3>
    <ul>
        <li>Total People: {self.session_total_people}</li>
        <li>Compliant: {self.session_compliant}</li>
        <li>Violations: {self.session_violations}</li>
    </ul>
    
    <h3 style="color: #f0ad4e;">Action Required:</h3>
    <p>Please ensure all personnel wear proper PPE equipment in the designated area.</p>
    
    <p style="color: #999; font-size: 12px; margin-top: 30px;">
        <em>Automated safety notification from PPE Detection System</em>
    </p>
</body>
</html>
"""

        # Send email asynchronously (non-blocking) with image
        if self.current_frame is not None:
            self.email_notifier.send_email_async(subject, body, self.current_frame)
            self.alert_log.append(f"[{timestamp}] Email notification sent (async)")

    def trigger_alert(self, violation_message):
        """Trigger audio and text alert with local timezone"""
        current_time = get_local_now()

        # Check alert cooldown
        if self.last_alert_time:
            time_diff = (current_time - self.last_alert_time).total_seconds()
            if time_diff < ALERT_COOLDOWN:
                return

        self.last_alert_time = current_time

        # Add to log with local timezone
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

        # Send Email notification
        self.send_email_notification(violation_message)

        # Update status
        self.status_label.setText(f"Status: VIOLATION DETECTED - {violation_message}")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")

    def update_frame(self):
        """Update video frame with detection - optimized for performance"""
        ret, frame = self.cap.read()

        if not ret:
            self.stop_detection()
            return

        self.frame_count += 1

        # Resize frame if too large for better performance
        frame_height, frame_width = frame.shape[:2]
        if frame_width > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / frame_width
            new_width = MAX_FRAME_WIDTH
            new_height = int(frame_height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            frame_height, frame_width = frame.shape[:2]

        # Store current frame (original without bounding boxes)
        original_frame = frame.copy()

        # Apply ROI if set
        detection_frame = frame.copy()
        roi_coords = None
        roi_offset = (0, 0)

        if self.use_roi:
            roi_coords = self.video_label.get_roi_coordinates(frame_width, frame_height)
            if roi_coords:
                x1, y1, x2, y2 = roi_coords
                detection_frame = frame[y1:y2, x1:x2]
                roi_offset = (x1, y1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "ROI", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Run detection
        if detection_frame.size > 0:
            results = self.model.predict(
                source=detection_frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            # Check for violations
            violation_results = self.check_violations(results)

            # Extract person bounding boxes for tracking
            person_bboxes = []
            for v_result in violation_results:
                bbox = v_result['bbox']
                # Convert to x, y, w, h format
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2] - bbox[0])
                h = int(bbox[3] - bbox[1])
                person_bboxes.append((x, y, w, h))

            # Update person tracker
            person_ids = []
            if self.person_tracker and len(person_bboxes) > 0:
                person_ids = self.person_tracker.update(person_bboxes)
            else:
                # If no tracker, assign sequential IDs
                person_ids = list(range(len(person_bboxes)))

            # Get annotated frame (with bounding boxes) BEFORE processing violations
            if self.use_roi and roi_coords:
                x1, y1, x2, y2 = roi_coords
                annotated_roi = results[0].plot()
                frame[y1:y2, x1:x2] = annotated_roi
                annotated_frame = frame
            else:
                annotated_frame = results[0].plot()

            # Update stored frame with annotations (for notifications with bounding boxes)
            self.current_frame = annotated_frame.copy()

            # Process each detected person
            for i, v_result in enumerate(violation_results):
                person_id = person_ids[i] if i < len(person_ids) else i

                # Log to database only once per person
                if person_id not in self.logged_person_ids:
                    self.logged_person_ids.add(person_id)
                    self.session_total_people += 1

                    if v_result['has_violation']:
                        self.session_violations += 1

                        # Save violation image (original frame without bounding boxes)
                        image_path = self.save_violation_image(original_frame, person_id)

                        # Log violation to database
                        self.db.log_violation(
                            CAMERA_ZONE_NAME,
                            v_result['violation_message'],
                            person_id,
                            v_result['confidence'],
                            image_path
                        )

                        # Trigger alert (will send annotated frame with bounding boxes)
                        self.trigger_alert(v_result['violation_message'])
                    else:
                        self.session_compliant += 1

                    # Log detection
                    self.db.log_detection(
                        CAMERA_ZONE_NAME,
                        person_id,
                        v_result['ppe_status'],
                        not v_result['has_violation']
                    )

                    # Update statistics display
                    self.update_statistics_display()

            # Reset status if no violations in current frame
            if not any(v['has_violation'] for v in violation_results):
                self.status_label.setText("Status: Detection running... No violations")
                self.status_label.setStyleSheet("")

            # Add violation warning overlay if any violations
            if any(v['has_violation'] for v in violation_results):
                cv2.putText(annotated_frame, "VIOLATION DETECTED!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # Draw person IDs if tracking enabled
            if self.person_tracker and len(person_ids) > 0:
                for i, person_id in enumerate(person_ids):
                    if i < len(person_bboxes):
                        x, y, w, h = person_bboxes[i]
                        # Adjust for ROI offset
                        x += roi_offset[0]
                        y += roi_offset[1]
                        cv2.putText(annotated_frame, f"ID:{person_id}", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
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