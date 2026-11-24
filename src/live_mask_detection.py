"""Run with: python -m src.live_mask_detection --model models/mask_detector.keras"""
import argparse
from pathlib import Path
import sys

import cv2

try:
    from .mask_detection import MaskClassifier
except ImportError:  # Also support python src/live_mask_detection.py.
    from mask_detection import MaskClassifier

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "mask_detector.keras"


def detect_faces(frame):
    """Preserve the existing half-resolution HOG detection approach."""
    import face_recognition
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    scale_x, scale_y = width / small.shape[1], height / small.shape[0]
    boxes, crops = [], []
    for top, right, bottom, left in face_recognition.face_locations(rgb, model="hog"):
        top, bottom = max(0, int(top * scale_y)), min(height, int(bottom * scale_y))
        left, right = max(0, int(left * scale_x)), min(width, int(right * scale_x))
        if bottom > top and right > left:
            boxes.append((top, right, bottom, left))
            crops.append(frame[top:bottom, left:right])
    return boxes, crops


def run(model_path, camera=0, labels=("MASK", "NO MASK")):
    classifier = MaskClassifier(model_path, labels)
    cap = None
    try:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera, backend)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot access camera {camera}. Check camera permissions and other camera apps.")
        print("Camera started. Press Q or close the window to quit.")
        window = "Live Mask Detection"
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError("Camera stopped returning frames. Check the camera connection and permissions.")
            boxes, crops = detect_faces(frame)
            results = classifier.classify(crops)
            for (top, right, bottom, left), (label, confidence) in zip(boxes, results):
                color = (0, 255, 0) if label == "MASK" else (0, 0, 255)
                text = f"{label}: {confidence:.1%}"
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                # Keep the detected box intact; clamp text at the image edge.
                box_top = top
                cv2.rectangle(frame, (left, box_top), (right, bottom), color, 2)
                x = max(0, min(left, frame.shape[1] - tw - 4))
                y = max(th + 4, box_top - baseline - 4)
                cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + baseline), (0, 0, 0), -1)
                cv2.putText(frame, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live MobileNetV2 mask classification")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mask-index", type=int, choices=(0, 1), default=0,
                        help="Output index for MASK; must match the model's training labels")
    args = parser.parse_args()
    labels = ("MASK", "NO MASK") if args.mask_index == 0 else ("NO MASK", "MASK")
    try:
        run(args.model, args.camera, labels)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
