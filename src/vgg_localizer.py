"""Fig. 1 VGG16 two-head architecture and deterministic volunteer-only training."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("COLUMNS", "160")  # Keep all Fig. 1 summary columns visible in saved logs.
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras

from .common import SEED, consent_subjects, digest, environment, local_path, setup, write_json

LOG = logging.getLogger(__name__)
EXPECTED_PARAMS = 16_826_181


@keras.utils.register_keras_serializable(package="VolunteerResearch")
def localization_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean squared xyxy error over positive examples; last target column is presence."""
    target = tf.cast(y_true[:, :4], y_pred.dtype)
    present = tf.cast(y_true[:, 4], y_pred.dtype)
    per_example = tf.reduce_mean(tf.square(target - y_pred), axis=-1)
    return tf.math.divide_no_nan(tf.reduce_sum(per_example * present), tf.reduce_sum(present))


def build_model(weights: str | None = "imagenet") -> keras.Model:
    """Match Fig. 1 names/shapes; the 512 pooling value is the output channel count."""
    inputs = keras.Input(shape=(120, 120, 3), name="input_layer_1")
    backbone = keras.applications.VGG16(include_top=False, weights=weights, input_shape=(120, 120, 3))
    backbone.trainable = True
    features = backbone(inputs)
    class_pool = keras.layers.GlobalMaxPooling2D(name="global_max_pooling2d")(features)
    bbox_pool = keras.layers.GlobalMaxPooling2D(name="global_max_pooling2d_1")(features)
    class_dense = keras.layers.Dense(2048, activation="relu", name="dense")(class_pool)
    bbox_dense = keras.layers.Dense(2048, activation="relu", name="dense_2")(bbox_pool)
    class_output = keras.layers.Dense(1, activation="sigmoid", name="dense_1")(class_dense)
    bbox_output = keras.layers.Dense(4, name="dense_3")(bbox_dense)
    model = keras.Model(inputs=inputs, outputs=[class_output, bbox_output], name="volunteer_localizer")
    total = model.count_params()
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    nontrainable = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    if (total, trainable, nontrainable) != (EXPECTED_PARAMS, EXPECTED_PARAMS, 0):
        raise AssertionError(f"Fig. 1 architecture mismatch: {total}, {trainable}, {nontrainable}")
    return model


def validate_box(box: list[float], width: int, height: int) -> np.ndarray:
    """Validate pixel xyxy annotations against the decoded image dimensions."""
    a = np.asarray(box, dtype=np.float32)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError("bbox must contain four finite pixel xyxy values")
    x1, y1, x2, y2 = a
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox outside {width}x{height} image or degenerate: {box}")
    return a


def augment(image: np.ndarray, bbox: np.ndarray, present: bool,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Crop, horizontal flip and rotate with synchronized pixel-space bbox transforms."""
    height, width = image.shape[:2]
    box = bbox.copy().astype(np.float32)
    crop_w, crop_h = int(width*rng.uniform(0.8, 1.0)), int(height*rng.uniform(0.8, 1.0))
    crop_w, crop_h = max(1, crop_w), max(1, crop_h)
    left, top = int(rng.integers(0, width-crop_w+1)), int(rng.integers(0, height-crop_h+1))
    cropped = np.array([max(box[0], left)-left, max(box[1], top)-top,
                        min(box[2], left+crop_w)-left, min(box[3], top+crop_h)-top], dtype=np.float32)
    intersection = max(0, cropped[2]-cropped[0]) * max(0, cropped[3]-cropped[1])
    original_area = max(1e-8, (box[2]-box[0]) * (box[3]-box[1]))
    if not present or intersection / original_area >= 0.8:
        image = image[top:top+crop_h, left:left+crop_w]
        if present:
            box = cropped
    height, width = image.shape[:2]
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
        if present:
            box = np.array([width-box[2], box[1], width-box[0], box[3]], dtype=np.float32)
    angle = float(rng.uniform(-15, 15))
    matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    if present:
        corners = np.array([[box[0], box[1], 1], [box[2], box[1], 1],
                            [box[2], box[3], 1], [box[0], box[3], 1]], dtype=np.float32)
        rotated = corners @ matrix.T
        box = np.array([np.clip(rotated[:, 0].min(), 0, width), np.clip(rotated[:, 1].min(), 0, height),
                        np.clip(rotated[:, 0].max(), 0, width), np.clip(rotated[:, 1].max(), 0, height)], dtype=np.float32)
    return image, box


def load_records(manifest: Path, subjects: set[str]) -> list[dict[str, Any]]:
    """Validate all training images before splitting; no silently discarded labels."""
    manifest = local_path(manifest)
    records = json.loads(manifest.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Training manifest must be a nonempty list")
    hashes: set[str] = set()
    for row in records:
        if row.get("consent") is not True:
            raise ValueError("Each training image needs consent=true, including background images")
        path = local_path(manifest.parent / row["path"])
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable training image: {path}")
        sha = digest(path)
        if sha in hashes:
            raise ValueError("Duplicate image content in training manifest")
        hashes.add(sha)
        row["sha256"], row["resolved_path"] = sha, str(path)
        if row.get("class") not in (0, 1) or type(row.get("class")) is not int:
            raise ValueError("class must be integer 0 (no face) or 1 (face present)")
        if not isinstance(row.get("session"), str) or not row["session"].strip():
            raise ValueError("Each image needs a capture-session identifier")
        if row["class"] == 1:
            if row.get("subject") not in subjects or row.get("condition") not in {"mask", "no_mask"}:
                raise ValueError("Positive images need a consented subject and mask/no_mask condition")
            validate_box(row["bbox"], image.shape[1], image.shape[0])
        elif row.get("subject") is not None or row.get("bbox") is not None:
            raise ValueError("Background images must have subject=null and bbox=null")
    return records


def split_records(records: list[dict[str, Any]], validation_fraction: float,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split connected subject/session groups, keeping volunteers and sessions disjoint."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    parents: dict[str, str] = {}
    def find(x: str) -> str:
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    for row in records:
        session = "session:" + row["session"]
        find(session)
        if row["class"]:
            subject = "subject:" + row["subject"]
            parents[find(subject)] = find(session)
    groups = [find("session:" + row["session"]) for row in records]
    if len(set(groups)) < 2:
        raise ValueError("Need at least two independent volunteer/session groups for a split")
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed).split(records, groups=groups))
    for label, indices in (("train", train), ("validation", validation)):
        rows = [records[int(i)] for i in indices]
        conditions = {r.get("condition") for r in rows if r["class"]}
        if not {"mask", "no_mask"} <= conditions or {r["class"] for r in rows} != {0, 1}:
            raise ValueError(f"{label} needs mask and no_mask positive images plus no-face negatives; add independent groups, do not search seeds")
    return train, validation


def make_dataset(records: list[dict[str, Any]], indices: np.ndarray, batch_size: int,
                 seed: int, training: bool) -> tf.data.Dataset:
    """Load real images and augment training data only, using a repeatable RNG stream."""
    rng = np.random.default_rng(seed)
    def generate():
        order = rng.permutation(indices) if training else indices
        for index in order:
            row = records[int(index)]
            image = cv2.imread(row["resolved_path"])
            if image is None:
                raise ValueError(f"Image became unreadable: {row['resolved_path']}")
            present = bool(row["class"])
            box = np.asarray(row["bbox"] if present else [0, 0, 0, 0], dtype=np.float32)
            if training:
                image, box = augment(image, box, present, rng)
            height, width = image.shape[:2]
            box = box / np.array([width, height, width, height], dtype=np.float32)
            image = cv2.cvtColor(cv2.resize(image, (120, 120)), cv2.COLOR_BGR2RGB).astype(np.float32)
            image = keras.applications.vgg16.preprocess_input(image)
            yield image, {"dense_1": np.array([present], dtype=np.float32),
                          "dense_3": np.concatenate([box, np.array([present], dtype=np.float32)])}
    signature = (tf.TensorSpec((120, 120, 3), tf.float32),
                 {"dense_1": tf.TensorSpec((1,), tf.float32), "dense_3": tf.TensorSpec((5,), tf.float32)})
    dataset = tf.data.Dataset.from_generator(generate, output_signature=signature).batch(batch_size)
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.apply(tf.data.experimental.assert_cardinality((len(indices) + batch_size - 1) // batch_size))
    return dataset.with_options(options)


def main() -> None:
    """Print the architecture or train and save real losses and a reproducible split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--consent", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/localizer"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    setup(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")
    if not args.summary and (args.manifest is None or args.consent is None):
        parser.error("Training requires --manifest and --consent")
    records = None
    if not args.summary:
        records = load_records(args.manifest, consent_subjects(args.consent))
        train, validation = split_records(records, args.validation_fraction, args.seed)
    model = build_model(None if args.weights == "none" else args.weights)
    model.summary(line_length=140)
    if args.summary:
        return
    args.output = local_path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output, 0o700)
    with (args.output / "model_summary.txt").open("w") as stream:
        model.summary(line_length=140, print_fn=lambda line: stream.write(line + "\n"))
    split = {"seed": args.seed, "requested_validation_fraction": args.validation_fraction,
             "train": [int(i) for i in train], "validation": [int(i) for i in validation],
             "records": records, "split_policy": "connected volunteer/session groups; no test set used in fit"}
    write_json(args.output / "split.json", split)
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate),
                  loss={"dense_1": keras.losses.BinaryCrossentropy(), "dense_3": localization_loss},
                  loss_weights={"dense_1": 1.0, "dense_3": 1.0})
    training = make_dataset(records, train, args.batch_size, args.seed, True)
    val = make_dataset(records, validation, args.batch_size, args.seed, False)
    history = model.fit(training, validation_data=val, epochs=args.epochs, verbose=2)
    model.save(args.output / "model.keras")
    write_json(args.output / "training.json", {"status": "completed", "seed": args.seed,
               "weights": args.weights, "epochs": args.epochs, "batch_size": args.batch_size,
               "learning_rate": args.learning_rate, "history": history.history,
               "environment": environment(), "manifest_sha256": digest(args.manifest),
               "consent_manifest_sha256": digest(args.consent)})


if __name__ == "__main__":
    main()
