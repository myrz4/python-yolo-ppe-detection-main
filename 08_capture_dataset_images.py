"""
Dataset Image Capture Tool
Continuously captures images from a camera and saves them for labeling (Roboflow) and training (Colab).

Examples:
    python 08_capture_dataset_images.py --person person01          # shows a folder menu
    python 08_capture_dataset_images.py --person person01 --condition 2
    python 08_capture_dataset_images.py --person person02 --condition no_helmet --interval 0.5
    python 08_capture_dataset_images.py --interval 0 --max-images 500   # every frame

Keys (preview window):
    1-8   = switch the save folder (see CONDITIONS below)
    SPACE = pause / resume auto capture
    S     = save one image immediately
    Q/ESC = quit
"""

import argparse
import os
import time
from datetime import datetime

import cv2

CONDITIONS = [
    "full_ppe",
    "no_helmet",
    "no_vest",
    "no_gloves",
    "no_goggles",
    "no_boots",
    "no_ppe",
    "holding_ppe",
]


def choose_condition(value):
    """Resolve a condition from a number (1-8), a known name, or an interactive menu."""
    if value:
        if value.isdigit() and 1 <= int(value) <= len(CONDITIONS):
            return CONDITIONS[int(value) - 1]
        return value

    print("Select the folder to save images to:")
    for i, name in enumerate(CONDITIONS, 1):
        print(f"  {i}. {name.replace('_', ' ')}")
    while True:
        choice = input(f"Enter 1-{len(CONDITIONS)}: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(CONDITIONS):
            return CONDITIONS[int(choice) - 1]
        print("Invalid choice, try again.")


def parse_args():
    p = argparse.ArgumentParser(description="Continuous camera image capture for dataset building")
    p.add_argument("--source", default="0", help="Camera index (0, 1, ...), video file, or RTSP URL")
    p.add_argument("--output", default="dataset_raw", help="Base output folder")
    p.add_argument("--person", default="person01", help="Person ID (used as folder name)")
    p.add_argument("--condition", default=None,
                   help="Folder: number 1-8 or a name (full_ppe, no_helmet, ...). Omit to choose from a menu")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Seconds between saved images. 0 = save every frame. 0.1 = ~10 images/second")
    p.add_argument("--max-images", type=int, default=0, help="Stop after N saved images (0 = unlimited)")
    p.add_argument("--duration", type=float, default=0, help="Stop after N seconds (0 = unlimited)")
    p.add_argument("--width", type=int, default=1280, help="Requested camera width")
    p.add_argument("--height", type=int, default=720, help="Requested camera height")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    p.add_argument("--min-sharpness", type=float, default=60.0,
                   help="Skip blurry frames below this Laplacian variance (0 = disable)")
    p.add_argument("--min-change", type=float, default=2.0,
                   help="Skip near-duplicate frames when mean pixel difference vs last saved is below this (0 = disable)")
    p.add_argument("--start-delay", type=float, default=3.0, help="Countdown seconds before capture starts")
    p.add_argument("--no-preview", action="store_true", help="Run without preview window")
    return p.parse_args()


def open_source(source, width, height):
    if not source.isdigit():
        return cv2.VideoCapture(source)

    index = int(source)
    backends = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY)] \
        if os.name == "nt" else [("default", cv2.CAP_ANY)]

    # Try each backend, first without then with a forced resolution, until real (non-black) frames arrive
    for name, backend in backends:
        for force_size in (False, True):
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            if force_size:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            for _ in range(60):  # camera warm-up: early frames are often black
                ok, frame = cap.read()
                if ok and frame is not None and frame.mean() > 5:
                    print(f"Camera ready (backend: {name}, forced size: {force_size})")
                    return cap
                time.sleep(0.03)
            cap.release()
    return cv2.VideoCapture(index)


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    args = parse_args()

    condition = choose_condition(args.condition)
    out_dir = os.path.join(args.output, args.person, condition)
    os.makedirs(out_dir, exist_ok=True)

    cap = open_source(args.source, args.width, args.height)
    if not cap.isOpened():
        print(f"Error: Cannot open source '{args.source}'. Try --source 1 or close other apps using the camera.")
        return

    print(f"Saving to: {os.path.abspath(out_dir)}")
    print("Keys: 1-8 = switch folder, SPACE = pause/resume, S = save one now, Q/ESC = quit")

    def set_condition(new_condition):
        nonlocal condition, out_dir, last_small, existing
        condition = new_condition
        out_dir = os.path.join(args.output, args.person, condition)
        os.makedirs(out_dir, exist_ok=True)
        last_small = None
        existing = len(os.listdir(out_dir))
        print(f"Switched to: {os.path.abspath(out_dir)}")

    saved = 0
    skipped_blur = 0
    skipped_dup = 0
    paused = False
    last_save_time = 0.0
    last_small = None
    start_time = time.time()
    capture_start = start_time + args.start_delay
    existing = len(os.listdir(out_dir))

    def save_frame(frame, force=False):
        nonlocal saved, skipped_blur, skipped_dup, last_small
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))

        if not force:
            if args.min_sharpness > 0 and sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < args.min_sharpness:
                skipped_blur += 1
                return False
            if args.min_change > 0 and last_small is not None:
                if cv2.absdiff(small, last_small).mean() < args.min_change:
                    skipped_dup += 1
                    return False

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        name = f"{args.person}_{condition}_{stamp}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        saved += 1
        last_small = small
        return True

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera/video ended or frame could not be read.")
            break

        now = time.time()
        counting_down = now < capture_start

        if not paused and not counting_down:
            if args.interval <= 0 or now - last_save_time >= args.interval:
                if save_frame(frame):
                    last_save_time = now

        if not args.no_preview:
            view = frame.copy()
            if counting_down:
                status = f"Starting in {capture_start - now:.0f}s..."
                color = (0, 255, 255)
            elif paused:
                status = "PAUSED"
                color = (0, 165, 255)
            else:
                status = "CAPTURING"
                color = (0, 255, 0)
            cv2.putText(view, f"{status} | {args.person}/{condition}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(view, f"Saved: {saved} (this session)  Blur skipped: {skipped_blur}  Dup skipped: {skipped_dup}",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(view, "Folder keys: " + " ".join(f"{i}={n}" for i, n in enumerate(CONDITIONS, 1)),
                        (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("Dataset Capture", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord(str(len(CONDITIONS))):
                set_condition(CONDITIONS[key - ord("1")])
            elif key == ord(" "):
                paused = not paused
            elif key == ord("s"):
                if save_frame(frame, force=True):
                    last_save_time = time.time()
        else:
            time.sleep(0.001)

        if args.max_images and saved >= args.max_images:
            print("Reached max images.")
            break
        if args.duration and now - capture_start >= args.duration and not counting_down:
            print("Reached duration limit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} new images in this session. "
          f"Skipped: {skipped_blur} blurry, {skipped_dup} duplicates.")
    print(f"Last folder: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
