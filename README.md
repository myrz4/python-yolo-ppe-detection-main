# YOLO PPE Detection System

A comprehensive Personal Protective Equipment (PPE) detection system using YOLOv11 for real-time safety monitoring in industrial environments.

## Features

- **Real-time PPE Detection**: Detects helmets, vests, gloves, boots, and goggles using YOLOv11
- **Multiple Video Sources**: Supports webcam, RTSP streams, and video files
- **Region of Interest (ROI)**: Define specific areas for monitoring
- **Person Tracking**: Track unique individuals across frames
- **Audio & Visual Alerts**: Real-time alerts for PPE violations
- **Telegram Notifications**: Automated notifications with violation images
- **Database Logging**: SQLite database for violation and detection logging
- **Analytics Dashboard**: Comprehensive reporting with charts and statistics
- **PDF Report Generation**: Export detailed compliance reports
- **Timezone Support**: All timestamps in Asia/Kuala_Lumpur timezone
- **Performance Optimized**: Frame resizing, async operations

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PPE Detection System                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Video Input → YOLO Detection → Person Tracking →           │
│                                                             │
│  → Violation Detection → Alerts/Notifications →             │
│                                                             │
│  → Database Logging → Analytics Dashboard                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.8 or higher
- Webcam or RTSP camera (for live detection)

## Model Training

Before setting up the detection system, you'll need a trained YOLO model for PPE detection.

### Option 1: Google Colab (Recommended)

We provide a ready-to-use Google Colab notebook for training your PPE detection model:

**[Open Training Notebook in Colab](https://colab.research.google.com/drive/1c654-sUZ-IdwwPlYaXQNtNysaOtMq3ir)**

Benefits:
- Free GPU access (Tesla T4, P100, or V100)
- Pre-configured environment
- Step-by-step training guide
- Automatic model export

### Option 2: Local Training

To train your own PPE detection model locally:

1. Prepare dataset with annotations (helmet, vest, gloves, boots, goggles, person)
2. Use YOLOv11 training:
   ```bash
   yolo train model=yolov11n.pt data=ppe_dataset.yaml epochs=100 imgsz=640
   ```
3. Place trained model in `models/best.pt`

### Training Tips

- **Dataset size**: Minimum 500-1000 images for good results
- **Annotations**: Use tools like Roboflow, LabelImg, or CVAT
- **Data augmentation**: Include various lighting conditions, angles, and backgrounds
- **Class balance**: Ensure balanced representation of all PPE classes
- **Validation split**: Use 80/20 train/validation split

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd python-yolo-ppe-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your YOLO model**
   - Place your trained YOLO model (`best.pt`) in the `models/` directory
   - You can train your own model or use a pre-trained PPE detection model

5. **Configure Telegram (Optional)**
   - Create a Telegram bot using [@BotFather](https://t.me/botfather) and get your bot token
   - Get your chat ID:
     - Send a message to your bot
     - Open in browser: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
     - Look for `"chat":{"id":123456789}` in the response
   - Update the credentials in the script files

## Scripts Overview

### 1. Basic Detection Scripts

#### `01_webcam_ppe_detection.py`
- Basic webcam PPE detection
- Simple and lightweight
- No GUI or logging

#### `02_rtsp_ppe_detection.py`
- RTSP stream support
- For IP cameras
- Command-line interface

### 2. GUI-Based Detection

#### `03_roi_ppe_detection.py`
- GUI interface with PySide6
- ROI (Region of Interest) selection
- Real-time visualization

#### `04_roi_alert_ppe_detection.py`
- Includes audio alerts
- Visual violation warnings
- Alert cooldown management

#### `05_roi_telegram_ppe_detection.py`
- Telegram notification integration
- Photo attachments with violations
- Configurable notification cooldown

### 3. Complete System

#### `06_roi_telegram_logging_ppe_detection.py`
The most complete detection system with:
- Full GUI interface
- ROI selection
- Audio alerts
- Telegram notifications (async)
- SQLite database logging
- Person tracking
- Session statistics
- **Timezone**: Asia/Kuala_Lumpur
- **Performance optimizations**:
  - Frame resizing for large inputs
  - Async Telegram notifications
  - Optimized YOLO inference

**Configuration Options**:
```python
# In the script file
TIMEZONE = 'Asia/Kuala_Lumpur'
CAMERA_ZONE_NAME = 'Main Entrance'
CONFIDENCE_THRESHOLD = 0.25
VIOLATION_MODE = 1  # 1: Any missing PPE, 2: Specific required PPE
MAX_FRAME_WIDTH = 1280  # Resize larger frames for performance
```

**Usage**:
```bash
python 06_roi_telegram_logging_ppe_detection.py
```

### 4. Analytics Dashboard

#### `07_ppe_reports_dashboard.py`
Comprehensive reporting dashboard with:
- Statistics overview (total, compliant, violations)
- Analytics charts (trends, violation types, hourly distribution)
- Violation log table
- Image gallery
- PDF report export
- Date range filtering
- Zone-based filtering
- **Timezone**: Asia/Kuala_Lumpur

**Usage**:
```bash
python 07_ppe_reports_dashboard.py
```

## Database Schema

### Tables

1. **violations**
   - `id`: Primary key
   - `timestamp`: Violation time (Asia/Kuala_Lumpur)
   - `zone_name`: Monitoring zone
   - `violation_type`: Description of violation
   - `person_id`: Tracked person ID
   - `confidence`: Detection confidence
   - `image_path`: Path to violation image

2. **detections**
   - `id`: Primary key
   - `timestamp`: Detection time (Asia/Kuala_Lumpur)
   - `zone_name`: Monitoring zone
   - `person_id`: Tracked person ID
   - `has_helmet`, `has_vest`, `has_gloves`, `has_boots`, `has_goggles`: Boolean flags
   - `is_compliant`: Compliance status

3. **daily_stats**
   - `id`: Primary key
   - `date`: Date (Asia/Kuala_Lumpur)
   - `zone_name`: Monitoring zone
   - `total_people`: Total people detected
   - `compliant_people`: Compliant count
   - `violation_people`: Violation count
   - `compliance_rate`: Percentage

## Performance Optimization

The system includes several optimizations to reduce lag:

1. **Frame Resizing**: Automatically resize large frames (controlled by `MAX_FRAME_WIDTH`)
2. **Async Telegram**: Non-blocking notifications using ThreadPoolExecutor
3. **Optimized YOLO**: Optimized inference settings
4. **Image Compression**: JPEG quality optimization for storage

### Performance Tips

- **For high-resolution cameras**: Set `MAX_FRAME_WIDTH = 1280` or lower
- **For slower systems**: Reduce `MAX_FRAME_WIDTH` to 960 or 640
- **For better quality**: Increase `MAX_FRAME_WIDTH` to 1920 (may impact performance)

## Configuration Guide

### Video Source Configuration

```python
# Webcam
VIDEO_SOURCE = 0

# RTSP Stream
VIDEO_SOURCE = 'rtsp://username:password@192.168.1.100:554/stream'

# Video File
VIDEO_SOURCE = 'path/to/video.mp4'
```

### Telegram Setup

1. **Create a Telegram Bot**
   - Open [@BotFather](https://t.me/botfather) on Telegram
   - Send `/newbot` and follow the instructions
   - Copy the bot token provided

2. **Get Your Chat ID**
   - Send any message to your newly created bot
   - Open this URL in your browser (replace `<YOUR_BOT_TOKEN>` with your actual token):
     ```
     https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
     ```
   - Look for `"chat":{"id":123456789}` in the JSON response
   - The number after `"id":` is your chat ID

3. **Update the Script**
   ```python
   TELEGRAM_BOT_TOKEN = 'your_bot_token_here'
   TELEGRAM_CHAT_ID = 'your_chat_id_here'
   ```

### Violation Detection Modes

**Mode 1: Any Missing PPE** (Strict)
- Triggers alert if ANY PPE is missing
- Recommended for high-risk areas

**Mode 2: Specific Required PPE** (Flexible)
- Only checks specific PPE items
- Configure via `REQUIRED_PPE = ['helmet', 'vest']`
- Recommended for areas with specific requirements

### Timezone Configuration

All timestamps use Asia/Kuala_Lumpur timezone:
```python
TIMEZONE = pytz.timezone('Asia/Kuala_Lumpur')
```

To change timezone, modify this line to your timezone (e.g., 'Asia/Singapore', 'UTC', etc.)

## Directory Structure

```
python-yolo-ppe-detection/
├── 01_webcam_ppe_detection.py
├── 02_rtsp_ppe_detection.py
├── 03_roi_ppe_detection.py
├── 04_roi_alert_ppe_detection.py
├── 05_roi_telegram_ppe_detection.py
├── 06_roi_telegram_logging_ppe_detection.py  ⭐
├── 07_ppe_reports_dashboard.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── best.pt                    # Your YOLO model
├── data/
│   ├── ppe_detection.db           # SQLite database
│   └── violations/                # Violation images
│       └── YYYY-MM-DD/
│           └── violation_*.jpg
└── .venv/                         # Virtual environment
```

## Usage Examples

### Basic Detection
```bash
python 01_webcam_ppe_detection.py
```

### Full System with Logging
```bash
python 06_roi_telegram_logging_ppe_detection.py
```

### View Reports
```bash
python 07_ppe_reports_dashboard.py
```

## Troubleshooting

### Telegram Not Working
- Verify bot token and chat ID
- Test bot with [@BotFather](https://t.me/botfather)
- Check internet connection

### Slow Performance
- Reduce `MAX_FRAME_WIDTH` to 960 or 640
- Use lower resolution camera
- Close other resource-intensive applications
- Consider using a simpler detection model

### Camera Not Displaying
- Ensure webcam is not being used by another application
- Try changing `VIDEO_SOURCE` to a different camera index (0, 1, 2, etc.)
- Check if model is loaded successfully (status bar message)
- Verify all dependencies are installed correctly

### Database Errors
- Ensure `data/` directory exists
- Check file permissions
- Delete and recreate database if corrupted

## Resources

- **Model Training**: [Google Colab Notebook](https://colab.research.google.com/drive/1c654-sUZ-IdwwPlYaXQNtNysaOtMq3ir)
- **YOLOv11**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **Telegram Bot Setup**:
  - Create bot: [@BotFather](https://t.me/botfather)
  - Get chat ID: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`

## Acknowledgments

- YOLOv11 by Ultralytics
- PySide6 for GUI framework
- OpenCV for computer vision
- Pygame for audio alerts
- ReportLab for PDF generation