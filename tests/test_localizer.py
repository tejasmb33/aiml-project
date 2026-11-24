"""Architecture, loss and bbox augmentation checks; no research metric claims."""
import numpy as np
import pytest
import tensorflow as tf

from src.vgg_localizer import EXPECTED_PARAMS, augment, build_model, localization_loss, split_records


def test_fig1_real_model_forward_and_gradient(tmp_path):
    tf.keras.utils.set_random_seed(2024)
    model = build_model(weights=None)  # Unit test stays offline; production defaults to ImageNet.
    assert model.count_params() == EXPECTED_PARAMS
    assert len(model.non_trainable_weights) == 0
    assert [layer.name for layer in model.layers] == ["input_layer_1", "vgg16", "global_max_pooling2d",
        "global_max_pooling2d_1", "dense", "dense_2", "dense_1", "dense_3"]
    inputs = tf.ones((1, 120, 120, 3))
    with tf.GradientTape() as tape:
        cls, bbox = model(inputs, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones((1, 1)), cls))
        loss += localization_loss(tf.constant([[0.1, 0.2, 0.8, 0.9, 1.0]]), bbox)
    gradients = tape.gradient(loss, model.trainable_weights)
    assert cls.shape == (1, 1) and bbox.shape == (1, 4)
    assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss={"dense_1": tf.keras.losses.BinaryCrossentropy(), "dense_3": localization_loss})
    targets = {"dense_1": tf.ones((1, 1)), "dense_3": tf.constant([[0.1, 0.2, 0.8, 0.9, 1.]])}
    losses = model.train_on_batch(inputs, targets)
    assert np.isfinite(losses).all()
    target = tmp_path / "test-only-untrained-fixture.keras"
    model.save(target)
    loaded = tf.keras.models.load_model(target)
    before, after = model(inputs), loaded(inputs)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-6)


def test_localization_loss_masks_negatives():
    target = tf.constant([[0., 0., 1., 1., 1.], [0., 0., 0., 0., 0.]])
    predicted = tf.constant([[0., 0., 0., 0.], [100., 100., 100., 100.]])
    assert float(localization_loss(target, predicted)) == pytest.approx(0.5)
    assert float(localization_loss(tf.zeros((2, 5)), predicted)) == 0


def test_augmentation_reproducible_and_boxes_valid():
    image = np.zeros((120, 160, 3), np.uint8)
    box = np.array([30, 20, 100, 90], np.float32)
    a, abox = augment(image, box, True, np.random.default_rng(2024))
    b, bbox = augment(image, box, True, np.random.default_rng(2024))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(abox, bbox)
    assert 0 <= abox[0] < abox[2] <= a.shape[1]
    assert 0 <= abox[1] < abox[3] <= a.shape[0]


def test_split_has_no_identity_or_session_leakage():
    records = []
    for subject in ["a", "b", "c", "d"]:
        for condition in ["mask", "no_mask"]:
            records.append({"subject": subject, "class": 1, "session": subject+"-session", "condition": condition})
        records.append({"subject": None, "class": 0, "session": subject+"-session"})
    train, val = split_records(records, 0.25, 2024)
    assert not {records[i]["session"] for i in train} & {records[i]["session"] for i in val}
    assert not {records[i]["subject"] for i in train if records[i]["class"]} & {records[i]["subject"] for i in val if records[i]["class"]}
    np.testing.assert_array_equal(train, split_records(records, 0.25, 2024)[0])


def test_single_connected_session_cannot_split():
    records = [{"subject": "a", "class": 1, "session": "same"}, {"subject": "b", "class": 1, "session": "same"}]
    with pytest.raises(ValueError, match="independent"):
        split_records(records, 0.25, 2024)
