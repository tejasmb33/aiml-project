"""Exercise real face_recognition distance/compare logic on numerical test vectors."""
import numpy as np
import pytest

from src.common import UNKNOWN
from src.recognize import Recognizer, detectCriminal, match_face


def vector(distance):
    result = np.zeros(128)
    result[0] = distance
    return result


def test_nearest_neighbor_real_distance():
    known = np.array([vector(0.5), vector(0.2)])
    match = match_face(vector(0), known, ["far", "near"], 0.6, 0.4)
    assert match.accepted and match.name == "near"
    assert match.distance == pytest.approx(0.2) and match.confidence == pytest.approx(0.8)


def test_both_thresholds_required():
    known = np.array([vector(0.5)])
    assert not match_face(vector(0), known, ["person"], 0.4, 0.1).accepted
    assert not match_face(vector(0), known, ["person"], 0.6, 0.7).accepted
    # Confidence comparison is strictly above, as requested.
    assert not match_face(vector(0), known, ["person"], 0.6, 0.5).accepted
    assert match_face(vector(0), known, ["person"], 0.5, 0.49).accepted


def test_empty_database_is_unknown():
    match = match_face(vector(0), np.empty((0, 128)), [])
    assert match.name == UNKNOWN and match.distance is None and not match.accepted


@pytest.mark.parametrize("tolerance,confidence", [(-1, 0.4), (float("nan"), 0.4), (0.6, 1.1), (0.6, -0.1), (0.6, float("nan"))])
def test_bad_thresholds_rejected(tolerance, confidence):
    with pytest.raises(ValueError):
        match_face(vector(0), np.array([vector(0)]), ["person"], tolerance, confidence)


def test_invalid_vectors_rejected():
    with pytest.raises(ValueError):
        match_face(np.zeros(127), np.zeros((1, 128)), ["person"])
    with pytest.raises(ValueError):
        match_face(np.full(128, np.nan), np.zeros((1, 128)), ["person"])


@pytest.mark.parametrize("detector", ["hog", "haar"])
def test_real_blank_frame_no_faces(detector):
    engine = Recognizer(np.empty((0, 128)), [], detector=detector)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    annotated, predictions = detectCriminal(frame, engine)
    assert predictions == []
    np.testing.assert_array_equal(frame, annotated)


def test_non_multiple_of_four_coordinates_and_label(monkeypatch):
    from src import recognize
    engine = Recognizer(np.zeros((1, 128)), ["asha"])
    frame = np.zeros((241, 321, 3), np.uint8)
    monkeypatch.setattr(recognize.face_recognition, "face_locations", lambda *a, **k: [(2, 20, 30, 1)])
    monkeypatch.setattr(recognize.face_recognition, "face_encodings", lambda *a, **k: [np.zeros(128)])
    annotated, predictions = detectCriminal(frame, engine)
    assert predictions[0]["bbox"] == [4, 8, 80, 120]
    assert predictions[0]["name"] == "asha"
    assert np.any(annotated[:, :, 1] > 0)
    assert not frame.any()
