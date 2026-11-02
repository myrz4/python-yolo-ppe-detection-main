"""
YOLO PPE Detection using Webcam
Simple webcam-based PPE detection system
"""

import cv2
from ultralytics import YOLO

# ==================== CONFIGURATION ====================
MODEL_PATH = 'models/best.pt'
CONFIDENCE_THRESHOLD = 0.25
WEBCAM_INDEX = 0  # 0 for default webcam, change if you have multiple cameras

# Class names
PPE_CLASSES = {
    'proper': ['helmet', 'gloves', 'vest', 'boots', 'goggles', 'Person'],
    'violations': ['no_helmet', 'no_goggle', 'no_gloves', 'no_boots', 'none']
}
# =======================================================


def main():
    # Load YOLO model
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Open webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"Error: Cannot open webcam {WEBCAM_INDEX}")
        return

    print("Starting PPE detection... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from webcam")
            break

        # Run YOLO detection
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Visualize results on frame
        annotated_frame = results[0].plot()

        # Display frame
        cv2.imshow('PPE Detection - Webcam', annotated_frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")


if __name__ == "__main__":
    main()
