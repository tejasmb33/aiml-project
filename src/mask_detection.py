"""MobileNetV2 mask inference; importing this module never opens a camera."""
from pathlib import Path

import cv2
import numpy as np


class MaskClassifier:
    """A trained 224x224 RGB MobileNetV2 with a two-class softmax output.

    Input preprocessing is performed here (pixels scaled to [-1, 1]). The saved
    model must not include a second preprocessing step. Labels follow training
    output order, which cannot be inferred from model weights.
    """

    def __init__(self, model_path, labels=("MASK", "NO MASK")):
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Mask model not found: {path}. Supply a trained MobileNetV2 "
                "classifier with --model; see models/README.md."
            )
        if tuple(labels) not in (("MASK", "NO MASK"), ("NO MASK", "MASK")):
            raise ValueError("Labels must contain MASK and NO MASK in training order")
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(str(path), compile=False)
        except Exception as exc:
            raise ValueError(f"Cannot load mask model {path}: {exc}") from exc
        if self.model.input_shape != (None, 224, 224, 3):
            raise ValueError("Model must accept (None, 224, 224, 3) RGB inputs")
        if self.model.output_shape != (None, 2):
            raise ValueError("Model must return (None, 2) softmax probabilities")
        def has_mobilenet(layer):
            return ("mobilenetv2" in layer.name.lower().replace("_", "")
                    or any(has_mobilenet(child) for child in getattr(layer, "layers", [])))
        if not has_mobilenet(self.model):
            raise ValueError("Model must contain a named MobileNetV2 backbone")
        self.labels = tuple(labels)

    def classify(self, bgr_faces):
        """Return (label, probability) for every crop in one inference batch."""
        if not bgr_faces:
            return []
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        batch = []
        for face in bgr_faces:
            if face is None or face.size == 0 or face.ndim != 3 or face.shape[2] != 3:
                raise ValueError("Each face must be a nonempty BGR image with 3 channels")
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            batch.append(cv2.resize(rgb, (224, 224)))
        inputs = preprocess_input(np.asarray(batch, dtype=np.float32))
        probabilities = np.asarray(self.model(inputs, training=False))
        if (probabilities.shape != (len(batch), 2)
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-3)):
            raise ValueError("Model must return two finite softmax probabilities per face")
        return [(self.labels[int(np.argmax(row))], float(np.max(row)))
                for row in probabilities]
