"""Measure recognition on annotated local volunteer frames/video; never synthesize results."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .common import SEED, UNKNOWN, consent_subjects, digest, environment, local_path, setup, write_json
from .recognize import add_recognition_args, detectCriminal, from_args

CONDITIONS = ("mask", "no_mask")
BACKGROUND = "__BACKGROUND__"
MISSED = "__MISSED__"
LOG = logging.getLogger(__name__)


def iou(a: list[float], b: list[float]) -> float:
    """Intersection over union of two xyxy boxes (exclusive right/bottom edges)."""
    intersection = max(0.0, min(a[2], b[2])-max(a[0], b[0])) * max(0.0, min(a[3], b[3])-max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def score_frame(truth: list[dict[str, Any]], predictions: list[dict[str, Any]],
                known_names: set[str], min_iou: float = 0.5) -> tuple[list[str], list[str]]:
    """One-to-one IoU matching penalizes missed faces, spurious faces and wrong IDs."""
    if not 0 < min_iou <= 1:
        raise ValueError("min_iou must be in (0, 1]")
    labels = [face["subject"] if face["subject"] in known_names else UNKNOWN for face in truth]
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    y_true: list[str] = []
    y_pred: list[str] = []
    if truth and predictions:
        overlaps = np.array([[iou(t["bbox"], p["bbox"]) for p in predictions] for t in truth])
        # Maximize the number of admissible matches first, then their IoU.
        admissible = overlaps >= min_iou
        utility = admissible * (min(len(truth), len(predictions)) + 1 + overlaps)
        rows, cols = linear_sum_assignment(-utility)
        for t, p in zip(rows.tolist(), cols.tolist()):
            if admissible[t, p]:
                matched_truth.add(t)
                matched_predictions.add(p)
                y_true.append(labels[t])
                y_pred.append(predictions[p]["name"])
    for t in range(len(truth)):
        if t not in matched_truth:
            y_true.append(labels[t])
            y_pred.append(MISSED)
    for p in range(len(predictions)):
        if p not in matched_predictions:
            y_true.append(BACKGROUND)
            y_pred.append(predictions[p]["name"])
    if not truth and not predictions:
        y_true.append(BACKGROUND)
        y_pred.append(BACKGROUND)
    return y_true, y_pred


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any] | None:
    """Macro metrics over observed semantic labels; MISSED remains a false negative."""
    if not y_true:
        return None
    labels = sorted((set(y_true) | set(y_pred)) - {MISSED})
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    matrix_labels = sorted(set(y_true) | set(y_pred))
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(np.mean(precision)), "recall_macro": float(np.mean(recall)),
            "f1_macro": float(np.mean(f1)), "scored_events": len(y_true),
            "per_class": {label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                          for label, p, r, f, s in zip(labels, precision, recall, f1, support)},
            "confusion_labels": matrix_labels,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=matrix_labels).tolist()}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Derive every summary from raw measured frames, with no target values."""
    if not rows:
        return None
    y_true = [v for row in rows for v in row["y_true"]]
    y_pred = [v for row in rows for v in row["y_pred"]]
    elapsed = np.array([row["processing_seconds"] for row in rows])
    fps = 1 / elapsed
    return {"frames": len(rows), "metrics": classification_metrics(y_true, y_pred),
            "mean_processing_seconds": float(elapsed.mean()),
            "processing_seconds_std": float(elapsed.std()),
            "aggregate_fps": float(len(rows)/elapsed.sum()),
            "mean_instantaneous_fps": float(fps.mean()), "fps_min": float(fps.min()), "fps_max": float(fps.max()),
            "efficiency_min": float(fps.min()*100), "efficiency_max": float(fps.max()*100)}


def plot_results(rows: list[dict[str, Any]], output: Path) -> None:
    """Paper titles/axes/colors; distinct line styles show real mask conditions."""
    specs = [("accuracy", "Face Recognition Accuracy Over Frames", "Recognition Accuracy (%)", "blue", (0, 40)),
             ("fps", "Face Recognition Speed Over Frames", "Speed (fps)", "green", (2, 25)),
             ("efficiency", "Face Recognition Efficiency Over Frames", "Efficiency", "red", (2, 25)),
             ("processing_seconds", "Face Recognition Processing Time Over Frames", "Processing Time (s)", "purple", (2, 25))]
    for key, title, ylabel, color, limits in specs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for condition, style in zip(CONDITIONS, ["-", "--"]):
            group = [row for row in rows if row["condition"] == condition]
            if not group:
                continue
            values = [r["cumulative_accuracy"]*100 if key == "accuracy" else r[key] for r in group]
            ax.plot([r["condition_index"] for r in group], values, color=color, linestyle=style, label=condition, marker=".")
        ax.set(title=title, xlabel="Frame Index", ylabel=ylabel, xlim=limits)
        if key == "accuracy":
            ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{key}.png", dpi=160)
        # Full sequence also retained if the paper window omits measured frames.
        if any(row["condition_index"] > limits[1] for row in rows):
            ax.set_xlim(0, max(row["condition_index"] for row in rows))
            fig.savefig(output / f"{key}_full.png", dpi=160)
        plt.close(fig)


def read_manifest(path: Path, subjects: set[str]) -> list[dict[str, Any]]:
    """Validate frame references/consent; image boxes are checked after decoding."""
    path = local_path(path)
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Evaluation manifest must be a nonempty list")
    seen: set[tuple[str, int | None]] = set()
    for row in records:
        if row.get("consent") is not True or row.get("condition") not in CONDITIONS:
            raise ValueError("Every frame needs consent=true and condition=mask/no_mask")
        if not isinstance(row.get("session"), str) or not row["session"].strip():
            raise ValueError("Every frame needs a capture-session identifier")
        if ("path" in row) == ("video" in row):
            raise ValueError("Specify exactly one of path (image) or video")
        if "video" in row and (type(row.get("frame_index")) is not int or row["frame_index"] < 0):
            raise ValueError("Video rows require a nonnegative integer frame_index")
        reference = local_path(path.parent / row.get("path", row.get("video")))
        if not reference.is_file():
            raise ValueError(f"Missing evaluation input: {reference}")
        key = (str(reference), row.get("frame_index") if "video" in row else None)
        if key in seen:
            raise ValueError("Duplicate evaluation frame reference")
        seen.add(key)
        row["resolved_path"] = str(reference)
        if not isinstance(row.get("faces"), list):
            raise ValueError("faces must be a list; use [] for a verified empty frame")
        for face in row["faces"]:
            if face.get("subject") not in subjects:
                raise ValueError("Every visible face needs a consented volunteer identifier")
    return records


def decode_frame(row: dict[str, Any]) -> np.ndarray:
    """Decode a genuine image/video frame, failing rather than skipping hard cases."""
    if "video" in row:
        capture = cv2.VideoCapture(row["resolved_path"])
        try:
            if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame_index"]):
                raise ValueError("Video cannot be opened/positioned")
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video frame cannot be decoded: {row['frame_index']}")
        finally:
            capture.release()
    else:
        frame = cv2.imread(row["resolved_path"])
        if frame is None:
            raise ValueError(f"Unreadable evaluation image: {row['resolved_path']}")
    for face in row["faces"]:
        box = np.asarray(face.get("bbox"), dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Each face needs a finite xyxy bbox")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
            raise ValueError("Evaluation bbox is outside frame or degenerate")
    return frame


def update_readme(path: Path, results: dict[str, Any]) -> None:
    """Replace only the explicit measured-results block with actual run summaries."""
    if not path.is_file():
        return
    text = path.read_text()
    start, end = "<!-- measured:start -->", "<!-- measured:end -->"
    if start not in text or end not in text:
        return
    lines = ["| Condition | Frames | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | FPS range | Efficiency range |",
             "|---|---:|---:|---:|---:|---:|---|---|"]
    for condition in CONDITIONS:
        summary = results["conditions"][condition]
        if summary is None:
            lines.append(f"| {condition} | — | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |")
        else:
            m = summary["metrics"]
            lines.append(f"| {condition} | {summary['frames']} | {m['accuracy']:.4f} | {m['precision_macro']:.4f} | {m['recall_macro']:.4f} | {m['f1_macro']:.4f} | {summary['fps_min']:.3f}–{summary['fps_max']:.3f} | {summary['efficiency_min']:.2f}–{summary['efficiency_max']:.2f} |")
    overall = results.get("overall")
    lines += ["", "| Quantity | Paper shows (approx.) | My measured |", "|---|---|---|"]
    measured = ["unavailable"]*3 if overall is None else [f"{overall['metrics']['accuracy']*100:.2f}% (whole-run, not a plateau)", f"{overall['fps_min']:.3f}–{overall['fps_max']:.3f}", f"{overall['efficiency_min']:.2f}–{overall['efficiency_max']:.2f}"]
    for label, paper, actual in zip(["Accuracy", "FPS", "Efficiency"], ["~80% plateau", "~1.1–3.3", "~110–290; formula unspecified"], measured):
        lines.append(f"| {label} | {paper} | {actual} |")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    path.write_text(before + start + "\n" + "\n".join(lines) + "\n" + end + after)


def main() -> None:
    """Evaluate supplied data, or write an explicit not-run report when none exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_recognition_args(parser)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--consent", type=Path)
    parser.add_argument("--training-split", type=Path, help="Reject overlap with recorded localizer train/validation data")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.manifest and not args.consent:
        parser.error("--manifest requires --consent")
    if not 0 < args.iou_threshold <= 1:
        parser.error("--iou-threshold must be in (0,1]")
    setup(args.seed)
    args.output = local_path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output, 0o700)
    # A fresh output directory prevents stale plots being mistaken for current results.
    if any(args.output.iterdir()):
        parser.error("Output directory must be empty; choose a new directory for each run")
    if args.manifest is None:
        results = {"status": "not_run", "reason": "No consented volunteer images, labels or recordings were supplied.",
                   "conditions": {c: None for c in CONDITIONS}, "overall": None,
                   "training_status": "not_run_no_volunteer_data", "screenshots": {}, "per_frame": [],
                   "environment": environment(), "seed": args.seed}
        write_json(args.output / "results.json", results)
        update_readme(args.readme, results)
        LOG.warning("No experiment run: report contains null metrics and no plots/screenshots")
        return
    subjects = consent_subjects(args.consent)
    records = read_manifest(args.manifest, subjects)
    recognizer = from_args(args)
    if not set(recognizer.names) <= subjects:
        raise ValueError("Enrollment database contains identifiers missing from the consent manifest")
    enrollment_report = args.encodings.with_suffix(".report.json")
    if not enrollment_report.is_file():
        raise ValueError("Enrollment provenance report is required; run src.encode first")
    enrollment = json.loads(enrollment_report.read_text())
    forbidden_hashes = {r["sha256"] for r in enrollment["images"]}
    forbidden_sessions: set[str] = set()
    if args.training_split:
        split = json.loads(args.training_split.read_text())
        forbidden_hashes.update(r["sha256"] for r in split["records"])
        forbidden_sessions.update(r["session"] for r in split["records"])
    hashes = {p: digest(Path(p)) for p in sorted({r["resolved_path"] for r in records})}
    if set(hashes.values()) & forbidden_hashes or any(r["session"] in forbidden_sessions for r in records):
        raise ValueError("Evaluation overlaps enrollment/training inputs or training sessions")
    rows = []
    screenshot_paths: dict[str, Any] = {}
    all_true: dict[str, list[str]] = {c: [] for c in CONDITIONS}
    all_pred: dict[str, list[str]] = {c: [] for c in CONDITIONS}
    counts = {c: 0 for c in CONDITIONS}
    for index, record in enumerate(records):
        frame = decode_frame(record)
        started = perf_counter()
        annotated, predictions = detectCriminal(frame, recognizer)
        elapsed = perf_counter() - started
        y_true, y_pred = score_frame(record["faces"], predictions, set(recognizer.names), args.iou_threshold)
        condition = record["condition"]
        all_true[condition].extend(y_true)
        all_pred[condition].extend(y_pred)
        row = {"manifest_index": index, "condition": condition, "condition_index": counts[condition],
               "source": record.get("path", record.get("video")), "source_frame_index": record.get("frame_index"),
               "session": record["session"], "source_sha256": hashes[record["resolved_path"]],
               "processing_seconds": elapsed, "fps": 1/elapsed, "efficiency": 100/elapsed,
               "y_true": y_true, "y_pred": y_pred, "predictions": predictions,
               "frame_accuracy": float(accuracy_score(y_true, y_pred)),
               "cumulative_accuracy": float(accuracy_score(all_true[condition], all_pred[condition]))}
        counts[condition] += 1
        rows.append(row)
        # First frame with a detection, regardless of success, prevents success cherry-picking.
        if condition not in screenshot_paths and predictions:
            target = args.output / f"recognition_{condition}.png"
            if not cv2.imwrite(str(target), annotated):
                raise RuntimeError("Screenshot write failed")
            os.chmod(target, 0o600)
            screenshot_paths[condition] = {"path": target.name, "manifest_index": index,
                                           "accepted_any": any(p["accepted"] for p in predictions)}
    results = {"status": "completed" if all(counts.values()) else "partial_missing_condition",
               "conditions": {c: summarize([r for r in rows if r["condition"] == c]) for c in CONDITIONS},
               "overall": summarize(rows), "per_frame": rows, "screenshots": screenshot_paths,
               "missing_screenshots": [c for c in CONDITIONS if c not in screenshot_paths],
               "seed": args.seed, "environment": environment(),
               "settings": {"detector": args.detector, "tolerance": args.tolerance,
                            "confidence_threshold": args.confidence_threshold, "iou_threshold": args.iou_threshold,
                            "resize_scale": 0.25, "efficiency_definition": "FPS * 100; not percent utilization",
                            "timing_scope": "resize + detection + encoding + matching + annotation; excludes decode, scoring, file IO and plots",
                            "warmup_frames": 0, "accuracy_curve": "cumulative event accuracy, separately per condition",
                            "frame_axis": "zero-based within-condition order in manifest"},
               "provenance": {"manifest_sha256": digest(args.manifest), "consent_sha256": digest(args.consent),
                              "encodings_sha256": digest(args.encodings), "enrollment_report_sha256": digest(enrollment_report),
                              "training_split_sha256": digest(args.training_split) if args.training_split else None}}
    write_json(args.output / "results.json", results)
    with (args.output / "frames.csv").open("w", newline="") as stream:
        fields = ["manifest_index", "condition", "condition_index", "processing_seconds", "fps", "efficiency", "frame_accuracy", "cumulative_accuracy"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot_results(rows, args.output)
    update_readme(args.readme, results)
    LOG.info("Measured %d frames. Results: %s", len(rows), args.output / "results.json")


if __name__ == "__main__":
    main()
