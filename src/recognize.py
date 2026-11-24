"""Webcam recognition and reusable frame processing, with optional Haar detection."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np

from .common import UNKNOWN, setup
from .encode import load_encodings


@dataclass(frozen=True)
class Match:
    """Nearest neighbor decision; confidence is a heuristic, not a probability."""
    name: str
    distance: float | None
    confidence: float | None
    accepted: bool


def validate_thresholds(tolerance: float, confidence_threshold: float) -> None:
    """Reject nonsensical/nonfinite thresholds rather than silently changing them."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")


def match_face(encoding: np.ndarray, known: np.ndarray, names: list[str],
               tolerance: float = 0.6, confidence_threshold: float = 0.4) -> Match:
    """Accept the nearest face iff compare_faces passes AND 1-distance > threshold."""
    validate_thresholds(tolerance, confidence_threshold)
    if known.shape != (len(names), 128) or np.asarray(encoding).shape != (128,):
        raise ValueError("Matcher requires aligned 128-dimensional arrays")
    if not np.isfinite(known).all() or not np.isfinite(encoding).all():
        raise ValueError("Encodings must be finite")
    if not names:
        return Match(UNKNOWN, None, None, False)
    distances = face_recognition.face_distance(known, encoding)
    comparisons = face_recognition.compare_faces(known, encoding, tolerance=tolerance)
    best = int(np.argmin(distances))
    distance = float(distances[best])
    confidence = 1.0 - distance
    accepted = bool(comparisons[best]) and confidence > confidence_threshold
    return Match(names[best] if accepted else UNKNOWN, distance, confidence, accepted)


class Recognizer:
    """One reusable detector/encoding database, shared by webcam and evaluation."""

    def __init__(self, known: np.ndarray, names: list[str], tolerance: float = 0.6,
                 confidence_threshold: float = 0.4, detector: str = "hog",
                 cascade_path: Path | None = None) -> None:
        validate_thresholds(tolerance, confidence_threshold)
        if detector not in {"hog", "cnn", "haar"}:
            raise ValueError("Detector must be hog, cnn or haar")
        self.known, self.names = known, names
        self.tolerance, self.confidence_threshold = tolerance, confidence_threshold
        self.detector = detector
        self.cascade = None
        if detector == "haar":
            path = cascade_path or Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(str(path))
            if self.cascade.empty():
                raise ValueError(f"Cannot load Haar cascade: {path}")

    def predict(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Quarter-size RGB encoding, returning full-size xyxy boxes and decisions."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 4:
            raise ValueError("Expected a BGR frame at least 4x4")
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (max(1, round(width * 0.25)), max(1, round(height * 0.25))))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if self.cascade is not None:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            rectangles = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            locations = [(int(y), int(x+w), int(y+h), int(x)) for x, y, w, h in rectangles]
        else:
            locations = face_recognition.face_locations(rgb, model=self.detector)
        vectors = face_recognition.face_encodings(rgb, known_face_locations=locations, num_jitters=1)
        if len(vectors) != len(locations):
            raise RuntimeError("Detector/encoder count mismatch")
        sx, sy = width / small.shape[1], height / small.shape[0]
        results = []
        for (top, right, bottom, left), vector in zip(locations, vectors):
            match = match_face(vector, self.known, self.names, self.tolerance, self.confidence_threshold)
            bbox = [max(0, round(left*sx)), max(0, round(top*sy)),
                    min(width, round(right*sx)), min(height, round(bottom*sy))]
            results.append({"bbox": bbox, "name": match.name, "distance": match.distance,
                            "confidence": match.confidence, "accepted": match.accepted})
        return results


def detectCriminal(frame: np.ndarray, recognizer: Recognizer,
                   predictions: list[dict[str, Any]] | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Draw paper-style green boxes and filled uppercase labels; never infer criminality."""
    predictions = recognizer.predict(frame) if predictions is None else predictions
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for result in predictions:
        x1, y1, x2, y2 = result["bbox"]
        label = result["name"].upper() if result["accepted"] else "UNKNOWN"
        cv2.rectangle(annotated, (x1, y1), (max(x1, x2-1), max(y1, y2-1)), (0, 255, 0), 2)
        font_scale = min(0.65, max(0.25, width / 1000))
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_top = min(max(0, y2 - text_height - baseline - 10), max(0, height - text_height - baseline - 10))
        label_right = min(width-1, max(x2, x1+text_width+8))
        cv2.rectangle(annotated, (x1, label_top), (label_right, min(height-1, label_top+text_height+baseline+10)), (0, 255, 0), cv2.FILLED)
        cv2.putText(annotated, label, (x1+4, label_top+text_height+4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated, predictions


def add_recognition_args(parser: argparse.ArgumentParser) -> None:
    """Keep evaluation and webcam settings identical."""
    parser.add_argument("--encodings", type=Path, default=Path("artifacts/encodings.npz"))
    parser.add_argument("--tolerance", type=float, default=0.6)
    parser.add_argument("--confidence-threshold", "--confidence_threshold", dest="confidence_threshold", type=float, default=0.4)
    parser.add_argument("--detector", choices=["hog", "cnn", "haar"], default="hog")
    parser.add_argument("--cascade", type=Path)


def from_args(args: argparse.Namespace) -> Recognizer:
    """Construct a reusable recognizer from common command line settings."""
    known, names = load_encodings(args.encodings)
    return Recognizer(known, names, args.tolerance, args.confidence_threshold, args.detector, args.cascade)


def main() -> None:
    """Run webcam recognition until q; camera/display resources always close."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_recognition_args(parser)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--consent-confirmed", action="store_true", help="Attest everyone in camera view consents")
    args = parser.parse_args()
    if not args.consent_confirmed:
        parser.error("Use --consent-confirmed only after everyone in camera view has consented")
    setup()
    recognizer = from_args(args)
    camera = cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam frame read failed")
            annotated, _ = detectCriminal(frame, recognizer)
            cv2.imshow("Volunteer Recognition - q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
