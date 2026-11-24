"""Synthetic test fixtures are software checks, never experiment data."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import encode


def image(path: Path) -> None:
    assert cv2.imwrite(str(path), np.zeros((64, 64, 3), dtype=np.uint8))


def test_real_blank_and_unreadable_are_warned(tmp_path, caplog):
    image(tmp_path / "blank.jpg")
    (tmp_path / "broken.jpg").write_bytes(b"invalid")
    vectors, names, report = encode.build_encodings(tmp_path, {"blank", "broken"})
    assert vectors.shape == (0, 128) and names == []
    assert all(row["status"] == "skipped" for row in report)
    assert "Unreadable" in caplog.text and "found 0" in caplog.text


def test_encoding_converts_bgr_and_keeps_filename(tmp_path, monkeypatch):
    pixels = np.zeros((20, 20, 3), np.uint8)
    pixels[:] = (10, 20, 30)
    cv2.imwrite(str(tmp_path / "asha.png"), pixels)
    def locations(rgb, model):
        np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])
        return [(0, 20, 20, 0)]
    monkeypatch.setattr(encode.face_recognition, "face_locations", locations)
    monkeypatch.setattr(encode.face_recognition, "face_encodings", lambda *a, **k: [np.arange(128)/128])
    vectors, names, report = encode.build_encodings(tmp_path, {"asha"})
    assert names == ["asha"] and vectors.shape == (1, 128)
    assert report[0]["status"] == "encoded"


@pytest.mark.parametrize("count", [0, 2])
def test_ambiguous_faces_skipped(tmp_path, monkeypatch, count):
    image(tmp_path / "person.png")
    monkeypatch.setattr(encode.face_recognition, "face_locations", lambda *a, **k: [(0, 10, 10, 0)]*count)
    vectors, names, _ = encode.build_encodings(tmp_path, {"person"})
    assert names == [] and vectors.shape == (0, 128)


def test_encoding_failure_is_logged(tmp_path, monkeypatch, caplog):
    image(tmp_path / "person.png")
    monkeypatch.setattr(encode.face_recognition, "face_locations", lambda *a, **k: [(0, 10, 10, 0)])
    def fail(*a, **k):
        raise RuntimeError("test encoder error")
    monkeypatch.setattr(encode.face_recognition, "face_encodings", fail)
    assert encode.build_encodings(tmp_path, {"person"})[1] == []
    assert "test encoder error" in caplog.text


def test_consent_and_duplicate_enrollment_fail(tmp_path):
    image(tmp_path / "person.jpg")
    with pytest.raises(ValueError, match="consent"):
        encode.build_encodings(tmp_path, set())
    image(tmp_path / "person.png")
    with pytest.raises(ValueError, match="More than one"):
        encode.build_encodings(tmp_path, {"person"})


def test_database_roundtrip_private_no_pickle(tmp_path):
    target = tmp_path / "faces.npz"
    vectors = np.zeros((2, 128))
    encode.save_encodings(target, vectors, ["one", "two"])
    actual, names = encode.load_encodings(target)
    np.testing.assert_array_equal(actual, vectors)
    assert names == ["one", "two"]
    assert target.stat().st_mode & 0o777 == 0o600


def test_invalid_database_rejected(tmp_path):
    target = tmp_path / "bad.npz"
    np.savez(target, encodings=np.zeros((1, 127)), names=np.array(["one"]))
    with pytest.raises(ValueError, match="Invalid encoding"):
        encode.load_encodings(target)
