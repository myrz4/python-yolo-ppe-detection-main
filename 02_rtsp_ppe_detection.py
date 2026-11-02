"""
YOLO PPE Detection using CCTV RTSP Stream
Real-time PPE detection from IP cameras via RTSP protocol
"""

import cv2
from ultralytics import YOLO

# ==================== CONFIGURATION ====================
MODEL_PATH = 'models/best.pt'
CONFIDENCE_THRESHOLD = 0.25

# RTSP Stream URL
# Format: rtsp://username:password@ip_address:port/stream_path
# Example: rtsp://admin:password123@192.168.1.100:554/stream1
RTSP_URL = 'rtsp://username:password@ip:port/stream'

# Connection settings
RTSP_TIMEOUT = 10  # seconds
RECONNECT_ATTEMPTS = 5

# Class names
PPE_CLASSES = {
    'proper': ['helmet', 'gloves', 'vest', 'boots', 'goggles', 'Person'],
    'violations': ['no_helmet', 'no_goggle', 'no_gloves', 'no_boots', 'none']
}
# =======================================================


def connect_to_stream(url, timeout=10):
    """Connect to RTSP stream with timeout"""
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

    # Set connection timeout
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)

    return cap


def main():
    # Load YOLO model
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Connect to RTSP stream
    print(f"Connecting to RTSP stream: {RTSP_URL}")
    cap = connect_to_stream(RTSP_URL, RTSP_TIMEOUT)

    if not cap.isOpened():
        print(f"Error: Cannot connect to RTSP stream {RTSP_URL}")
        print("Please check:")
        print("  - RTSP URL format")
        print("  - Username and password")
        print("  - Network connectivity")
        print("  - Camera is online and streaming")
        return

    print("Connected successfully! Starting PPE detection... Press 'q' to quit")

    frame_count = 0
    reconnect_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"Error: Lost connection to stream (attempt {reconnect_count + 1}/{RECONNECT_ATTEMPTS})")

            # Attempt reconnection
            if reconnect_count < RECONNECT_ATTEMPTS:
                reconnect_count += 1
                cap.release()
                print("Attempting to reconnect...")
                cap = connect_to_stream(RTSP_URL, RTSP_TIMEOUT)

                if cap.isOpened():
                    print("Reconnected successfully!")
                    reconnect_count = 0
                    continue
                else:
                    print(f"Reconnection failed. Waiting 2 seconds...")
                    cv2.waitKey(2000)
                    continue
            else:
                print("Max reconnection attempts reached. Exiting...")
                break

        # Reset reconnect counter on successful frame read
        reconnect_count = 0
        frame_count += 1

        # Run YOLO detection
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Visualize results on frame
        annotated_frame = results[0].plot()

        # Add stream info overlay
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('PPE Detection - RTSP Stream', annotated_frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")


if __name__ == "__main__":
    main()
