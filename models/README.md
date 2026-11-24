# Mask detection model

No trained mask weights are bundled. The live face detection script is unchanged.
Generic ImageNet MobileNetV2 weights do **not** classify masks; a trained mask
classification head is required. No accuracy or live mask results are claimed.

Use the existing environment:

```sh
source .venv/bin/activate
python -m src.live_mask_detection --model models/mask_detector.keras
```

Use `--camera 1` for another camera. Press Q or close the window to exit.
On macOS the runner uses AVFoundation, as the working face script does. Enable
camera access for your terminal/IDE in System Settings if access fails.
No additional dependency installation or Metal plugin is required by this module;
it uses the installed TensorFlow/Keras and can run on CPU on Apple Silicon.

## Model contract

Supply a trusted, locally saved Keras `.keras` or `.h5` **full model**, not a
weights-only file. It must contain a named MobileNetV2 backbone (keep the default
Keras MobileNetV2 name), accept `(None, 224, 224, 3)`, and return `(None, 2)`
softmax probabilities. It must have been trained for mask/no-mask classification.

The runner converts OpenCV BGR crops to RGB, resizes to 224x224, and applies
`tensorflow.keras.applications.mobilenet_v2.preprocess_input` ([-1, 1]). Do not
include that preprocessing a second time inside the saved model. Training must
use the same preprocessing.

Default output order is `[MASK, NO MASK]`. Check your training class mapping;
use `--mask-index 1` if the order is `[NO MASK, MASK]`. The weights alone cannot
reveal this mapping. Sigmoid and logits outputs are intentionally rejected.
The displayed confidence is the selected softmax score, not a calibrated
probability of correctness.

All valid detected crops are classified in one batch. HOG detection retains the
existing approach and may miss occluded or masked faces; it needs real webcam
validation. Test with masked/unmasked people and multiple faces once the trained
model is supplied. Record the weights' source, license, training label order,
and evaluation results alongside the model before claiming detection accuracy.
