"""Metric checks with explicit fixtures, separate from volunteer research results."""
import numpy as np
import pytest

from src.common import UNKNOWN
from src.evaluate import BACKGROUND, MISSED, classification_metrics, iou, plot_results, score_frame, summarize


def face(subject="asha", box=None):
    return {"subject": subject, "bbox": box or [0, 0, 10, 10]}


def pred(name="asha", box=None):
    return {"name": name, "bbox": box or [0, 0, 10, 10]}


def test_iou_and_missed_extra_wrong_id():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert score_frame([face()], [], {"asha"}) == (["asha"], [MISSED])
    assert score_frame([], [pred()], {"asha"}) == ([BACKGROUND], ["asha"])
    assert score_frame([face()], [pred("other")], {"asha"}) == (["asha"], ["other"])
    assert score_frame([], [], {"asha"}) == ([BACKGROUND], [BACKGROUND])
    assert score_frame([face("unenrolled")], [pred(UNKNOWN)], {"asha"}) == ([UNKNOWN], [UNKNOWN])


def test_each_prediction_used_once():
    true, predicted = score_frame([face(), face()], [pred()], {"asha"})
    assert true == ["asha", "asha"] and sorted(predicted) == sorted(["asha", MISSED])


def test_metrics_no_invented_success():
    metrics = classification_metrics(["a", "a", "b", "b"], ["a", MISSED, "b", "a"])
    assert metrics["accuracy"] == 0.5
    assert metrics["precision_macro"] == pytest.approx(0.75)
    assert metrics["recall_macro"] == 0.5
    assert metrics["f1_macro"] == pytest.approx((0.5+2/3)/2)
    assert classification_metrics([], []) is None


def test_plots_and_summaries_derive_from_rows(tmp_path):
    rows = [{"condition": "no_mask", "condition_index": i, "y_true": ["a"], "y_pred": ["a"],
             "processing_seconds": sec, "fps": 1/sec, "efficiency": 100/sec, "cumulative_accuracy": 1.0}
            for i, sec in enumerate([0.5, 1.0, 2.0])]
    summary = summarize(rows)
    assert summary["aggregate_fps"] == pytest.approx(3/3.5)
    assert summary["fps_min"] == 0.5 and summary["fps_max"] == 2
    assert summary["efficiency_min"] == 50 and summary["efficiency_max"] == 200
    plot_results(rows, tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_real_cli_with_synthetic_backgrounds_only(tmp_path, monkeypatch):
    """Integration check only: no people, face identities or research results."""
    import json
    import sys
    import cv2
    from src import evaluate
    from src.encode import save_encodings
    consent = tmp_path / "consent.json"
    consent.write_text(json.dumps({"attested": True, "subjects": ["synthetic_test_identifier"]}))
    database = tmp_path / "encodings.npz"
    save_encodings(database, np.zeros((1, 128)), ["synthetic_test_identifier"])
    database.with_suffix(".report.json").write_text(json.dumps({"images": []}))
    records = []
    for condition, value in [("mask", 0), ("no_mask", 255)]:
        image = tmp_path / (condition + ".png")
        cv2.imwrite(str(image), np.full((240, 320, 3), value, np.uint8))
        records.append({"path": image.name, "condition": condition, "session": "synthetic-test",
                        "consent": True, "faces": []})
    manifest = tmp_path / "test.json"
    manifest.write_text(json.dumps(records))
    output = tmp_path / "test-output"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--manifest", str(manifest), "--consent", str(consent),
                                     "--encodings", str(database), "--output", str(output),
                                     "--readme", str(tmp_path / "no-readme")])
    evaluate.main()
    result = json.loads((output / "results.json").read_text())
    assert result["status"] == "completed"
    assert len(result["per_frame"]) == 2 and result["screenshots"] == {}
    assert set(result["missing_screenshots"]) == {"mask", "no_mask"}
    assert all(r["processing_seconds"] > 0 for r in result["per_frame"])
    assert all(r["y_true"] == [BACKGROUND] for r in result["per_frame"])
    assert len(list(output.glob("*.png"))) == 4
    with pytest.raises(SystemExit):
        evaluate.main()  # stale output directories are rejected
