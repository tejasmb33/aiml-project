"""Shared reproducibility, consent, provenance and serialization helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

SEED = 2024
UNKNOWN = "__UNKNOWN__"
RESERVED = {UNKNOWN, "__BACKGROUND__", "__MISSED__"}


def setup(seed: int = SEED) -> None:
    """Seed Python, NumPy and OpenCV; limit CPU parallelism for repeatability."""
    import cv2
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(1)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def local_path(value: str | Path) -> Path:
    """Reject network URLs: all biometric inputs and outputs must be local."""
    if "://" in str(value):
        raise ValueError("Only local filesystem paths are supported")
    return Path(value).expanduser().resolve()


def consent_subjects(path: Path) -> set[str]:
    """Read an operator attestation; this does not verify legal consent."""
    data = json.loads(local_path(path).read_text())
    subjects = data.get("subjects", [])
    if data.get("attested") is not True or not subjects:
        raise ValueError("Consent manifest must attest consent and list volunteer subjects")
    if any(not isinstance(s, str) or not s.strip() or s in RESERVED for s in subjects):
        raise ValueError("Invalid/reserved volunteer identifier")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Duplicate consent subject identifiers")
    return set(subjects)


def digest(path: Path) -> str:
    """Compute an input hash without exposing image contents."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any, private: bool = True) -> None:
    """Write strict JSON atomically, with private permissions by default."""
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    if private:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.chmod(temp, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def environment() -> dict[str, Any]:
    """Describe the machine and installed versions used for a real run."""
    packages = {}
    for package in ("numpy", "opencv-python", "dlib", "face-recognition", "tensorflow",
                    "keras", "scikit-learn", "matplotlib", "scipy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if not cpu_model and cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": cpu_model, "cpu_count": os.cpu_count(),
            "packages": packages}
