from unittest.mock import Mock

import numpy as np
import pytest

from src.mask_detection import MaskClassifier
from src import live_mask_detection as live


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Supply a trained MobileNetV2"):
        MaskClassifier(tmp_path / "absent.keras")


def classifier_with_output(output):
    classifier = MaskClassifier.__new__(MaskClassifier)
    classifier.labels = ("MASK", "NO MASK")
    classifier.model = Mock(return_value=np.asarray(output))
    return classifier


def test_batch_preprocessing_and_labels():
    classifier = classifier_with_output([[0.9, 0.1], [0.2, 0.8]])
    blue = np.full((20, 30, 3), [255, 0, 0], dtype=np.uint8)
    results = classifier.classify([blue, blue])
    assert results[0] == ("MASK", 0.9)
    assert results[1] == ("NO MASK", 0.8)
    batch = classifier.model.call_args.args[0]
    assert batch.shape == (2, 224, 224, 3)
    np.testing.assert_allclose(batch[0, 0, 0], [-1, -1, 1])


@pytest.mark.parametrize("output", [[[2, -1]], [[float("nan"), 0]], [[0.2, 0.2]], [[1]]])
def test_reject_invalid_probabilities(output):
    with pytest.raises(ValueError, match="softmax"):
        classifier_with_output(output).classify([np.zeros((10, 10, 3), np.uint8)])


def test_empty_batch_skips_inference():
    classifier = classifier_with_output([])
    assert classifier.classify([]) == []
    classifier.model.assert_not_called()


@pytest.mark.parametrize("opened, frame", [(False, None), (True, None)])
def test_camera_failure_releases_resources(monkeypatch, opened, frame):
    cap = Mock()
    cap.isOpened.return_value = opened
    cap.read.return_value = (False, frame)
    cleanup = Mock()
    monkeypatch.setattr(live, "MaskClassifier", Mock())
    monkeypatch.setattr(live.cv2, "VideoCapture", Mock(return_value=cap))
    monkeypatch.setattr(live.cv2, "destroyAllWindows", cleanup)
    with pytest.raises(RuntimeError, match="camera|Camera"):
        live.run("unused.keras")
    cap.release.assert_called_once()
    cleanup.assert_called_once()
