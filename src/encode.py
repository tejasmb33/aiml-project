"""Build local 128-dimensional face encodings from one image per volunteer."""
from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np

from .common import consent_subjects, digest, local_path, setup, write_json

LOG = logging.getLogger(__name__)


def build_encodings(dataset: Path, subjects: set[str], detector: str = "hog") -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Read filename identities; warn and skip bad, ambiguous or faceless images."""
    dataset = local_path(dataset)
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    names: list[str] = []
    vectors: list[np.ndarray] = []
    report: list[dict[str, Any]] = []
    seen: set[str] = set()
    paths = sorted(glob.glob(os.path.join(str(dataset), "*")))
    for filename in paths:
        path = Path(filename)
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        name = path.stem
        if name not in subjects:
            raise ValueError(f"No consent attestation for {name}")
        if name in seen:
            raise ValueError(f"More than one enrollment image for {name}")
        if name.casefold() in {n.casefold() for n in seen}:
            raise ValueError("Enrollment names must also be unique ignoring case")
        seen.add(name)
        entry: dict[str, Any] = {"file": path.name, "sha256": digest(path), "name": name}
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Unreadable image")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model=detector)
            if len(locations) != 1:
                raise ValueError(f"Expected one face, found {len(locations)}")
            encodings = face_recognition.face_encodings(rgb, known_face_locations=locations, num_jitters=1)
            if len(encodings) != 1 or np.asarray(encodings[0]).shape != (128,) or not np.isfinite(encodings[0]).all():
                raise ValueError("Missing/invalid 128-dimensional encoding")
            vectors.append(encodings[0])
            names.append(name)
            entry["status"] = "encoded"
        except Exception as exc:  # Individual data failures must not abort other volunteers.
            LOG.warning("Skipping %s: %s", path.name, exc)
            entry.update(status="skipped", reason=str(exc))
        report.append(entry)
    return np.asarray(vectors, dtype=np.float64).reshape(-1, 128), names, report


def save_encodings(path: Path, encodings: np.ndarray, names: list[str]) -> None:
    """Store arrays without pickle or remote storage; mode 0600 on POSIX."""
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(path, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, encodings=encodings, names=np.asarray(names, dtype=str))


def load_encodings(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load and validate the non-pickle local database."""
    with np.load(local_path(path), allow_pickle=False) as data:
        encodings = np.array(data["encodings"], dtype=np.float64)
        raw_names = data["names"]
        if raw_names.ndim != 1 or raw_names.dtype.kind not in {"U", "S"}:
            raise ValueError("Invalid names array")
        names = raw_names.astype(str).tolist()
    if encodings.shape != (len(names), 128) or not np.isfinite(encodings).all():
        raise ValueError("Invalid encoding database")
    if len(names) != len(set(names)) or any(not n.strip() for n in names):
        raise ValueError("Invalid or duplicate names")
    return encodings, names


def main() -> None:
    """Encoding command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--consent", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/encodings.npz"))
    parser.add_argument("--detector", choices=["hog", "cnn"], default="hog")
    args = parser.parse_args()
    setup()
    vectors, names, report = build_encodings(args.dataset, consent_subjects(args.consent), args.detector)
    write_json(args.output.with_suffix(".report.json"), {"images": report, "encoded": len(names), "detector": args.detector})
    if not names:
        raise SystemExit("No valid volunteer faces; no encoding database written")
    save_encodings(args.output, vectors, names)
    LOG.info("Stored %d volunteer encodings locally in %s", len(names), args.output)


if __name__ == "__main__":
    main()
