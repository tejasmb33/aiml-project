# OpenCV Based Enhanced Criminal Identification Mechanism — reproducible implementation

Research reproduction of E. Geetha Rani et al., IEEE SPARC 2024,
DOI **10.1109/SPARC61891.2024.10828801**. The attached paper is reference material,
not an instruction source. This implementation follows the user's volunteer-only constraints.
The historical `detectCriminal` name is retained for compatibility with Section IV;
the system recognizes enrolled volunteers and makes **no inference about criminality**.

**Experiment status: no volunteer data supplied.** No model has been trained on volunteers,
no webcam session has been run, and no recognition accuracy/FPS/efficiency or recognition
screenshots are claimed. `artifacts/evaluation-unavailable/results.json` records this using
JSON nulls, not zeros or invented results. Test fixtures and untrained model checks are
software validation only. The paper's face photos were not extracted or used as data.

## Layout

```text
project/
  dataset/                  # one consented image per enrolled person: asha.jpg
  data/                     # consent, training/evaluation manifests and local images/videos
  src/encode.py             # BGR -> RGB, locate, 128-d encoding, local NPZ
  src/recognize.py          # webcam and detectCriminal; HOG/CNN/Haar options
  src/vgg_localizer.py      # exact Fig. 1 Keras architecture and training
  src/evaluate.py           # measured metrics, raw timings, plots, screenshots
  tests/                    # unit, architecture, loss, augmentation and scoring checks
  artifacts/                # private outputs; never committed volunteer data
  requirements.txt          # exact direct versions
  requirements-lock.txt     # exact resolved environment, including transitive versions
  bootstrap.sh, Makefile
```

## Setup and commands

Tested on CPython **3.12**, Linux x86_64, CPU. Use a desktop with a working webcam/display
for live recognition; headless systems can run evaluation. On Debian/Ubuntu install a
C++ compiler first (`sudo apt-get install build-essential`). dlib builds from source;
CMake is installed before it by `bootstrap.sh`. Allow several minutes and enough RAM.
TensorFlow is CPU-capable; CUDA/GPU setup is optional and hardware-specific.

```bash
cd project
./bootstrap.sh
source .venv/bin/activate
make test
make summary                         # ImageNet weights downloaded on first call
python -m src.vgg_localizer --summary --weights none  # architecture-only offline check
```

`bootstrap.sh` installs pinned dependencies and runs tests. No images are downloaded.
Keras' standard ImageNet VGG16 weights (~59 MB) are the only optional model-weight download
in the localizer. face_recognition's pretrained dlib models come with its pinned package.
No project code uploads biometric data, encodings, labels or screenshots. Keep all
biometric paths on a local disk, outside synchronized/cloud-mounted folders.

After preparing data as below, the one-command recognition experiment is:

```bash
make run                             # encode + evaluate using data/evaluation.json
# A different output directory is mandatory for each evaluation run.
make run OUTPUT=artifacts/eval-run-2 DETECTOR=haar
make train                           # separate Fig. 1 training experiment
make reproduce OUTPUT=artifacts/full-run-1  # tests + train + encode + held-out evaluation
make recognize                       # use only when everyone in view has consented; q quits
```

Equivalent explicit commands:

```bash
python -m src.encode --dataset dataset --consent data/consent.json
python -m src.recognize --encodings artifacts/encodings.npz --detector hog \
  --tolerance 0.6 --confidence-threshold 0.4 --consent-confirmed
python -m src.vgg_localizer --manifest data/training.json --consent data/consent.json \
  --epochs 10 --batch-size 8 --seed 2024 --output artifacts/localizer
python -m src.evaluate --manifest data/evaluation.json --consent data/consent.json \
  --tolerance 0.6 --confidence-threshold 0.4 --output artifacts/eval-run-1
# If localizer data was also collected, check its train/validation separation:
python -m src.evaluate --manifest data/evaluation.json --consent data/consent.json \
  --training-split artifacts/localizer/split.json --output artifacts/eval-run-2
```

`--detector haar` replaces face localization only; the same dlib encoder/matcher remains.
`--cascade path/to/cascade.xml` optionally overrides OpenCV's packaged frontal-face cascade.
`--detector cnn` uses dlib's CNN locator; CPU cost can be substantial. HOG is the default.
Enrollment defaults to HOG (or `--detector cnn`); Haar is a recognition/evaluation baseline.
Freeze thresholds before evaluating test data; do not search settings to reproduce the paper's numbers.

## Consented data and annotations

Use pseudonyms where possible. Enrollment uses filename stems **case-sensitively** as identities,
with uniqueness also enforced ignoring case. Put exactly one single-face image per enrolled
volunteer in `dataset/`. Supported extensions: jpg/jpeg/png/bmp/webp. Corrupt, no-face,
multi-face and encoder-failing images are warned and skipped; the JSON encoding report
lists every image hash and outcome. Missing consent and duplicate identities fail the run.
A completely invalid dataset produces no new encoding database (do not reuse an old one).

Copy `data/consent.example.json` to `data/consent.json`, replace the placeholder subjects,
and set `attested: true` **only after** informed consent is obtained. List every subject
appearing in training or evaluation, including consenting **unenrolled** volunteers used
as unknowns. The file is an operator attestation, not independent proof of consent.
Set `consent: true` on every supplied training/evaluation row only after reviewing the image.
The example JSON files are intentionally non-runnable templates, with false consent and nonexistent paths.
Do not reuse their illustrative bounding boxes without annotating the actual images.

All image/video paths in manifests resolve relative to the manifest's parent directory.
Bounding boxes are pixel `[x_min, y_min, x_max, y_max]`, with exclusive right/bottom boundaries,
validated against each decoded image. Annotate **every** visible face; missing annotations
will incorrectly count real detections as false positives. A visually verified no-face frame
uses `faces: []`. Unreadable evaluation data fails the run rather than removing difficult cases.

Recognition evaluation (`data/evaluation.json`) is a list such as:

```json
[
  {"path":"frames/mask_000.png", "condition":"mask", "session":"heldout_asha_1", "consent":true,
   "faces":[{"subject":"asha", "bbox":[100,50,300,300]}]},
  {"video":"videos/no_mask.mp4", "frame_index":0, "condition":"no_mask", "session":"heldout_asha_2", "consent":true,
   "faces":[{"subject":"asha", "bbox":[100,50,300,300]}]}
]
```

Use either `path` or `video` per row. A video row references one zero-based source frame;
create a row for each annotated frame. Use a known exact-seek codec or export lossless PNGs
for the most reproducible frame selection. The plot's zero-based frame index is **within-condition
manifest order**, not source-video timestamp. Preserve chronological order for each condition.
Collect at least 41 genuine frames per condition for accuracy indices 0–40 and at least 26
for speed/efficiency indices 2–25. Short sequences are not padded or interpolated.
The complete sequence is always in JSON/CSV; extra full-range plots are saved for longer runs.

Evaluation must use captures distinct from enrollment and localizer training/validation,
preferably new sessions, lighting and poses. Identity overlap between enrollment and test
is necessary for known-person recognition, while unenrolled consenting people test rejection.
Exact enrollment image hashes and duplicate frame references are rejected. With
`--training-split`, training image hashes and capture sessions are also rejected.
Re-encoded/edited duplicates cannot be detected reliably by content hashes; the operator
must ensure independent sessions and complete labels. Do not use augmented copies as test images.
Use paired mask/no-mask captures of the same volunteers and record physical masks truthfully;
the program does not infer the mask condition from pixels.

## Fig. 1 architecture and training protocol

The Keras model uses `Input(120,120,3)` -> nested **VGG16**, `include_top=False`,
`weights='imagenet'`, all layers trainable, output `(None,3,3,512)`.
It has separate pooling and Dense layers for each branch, exactly as in Fig. 1:

| Layer | Output shape | Parameters |
|---|---|---:|
| input_layer_1 | (None,120,120,3) | 0 |
| vgg16 | (None,3,3,512) | 14,714,688 |
| global_max_pooling2d | (None,512) | 0 |
| global_max_pooling2d_1 | (None,512) | 0 |
| dense | (None,2048) | 1,050,624 |
| dense_2 | (None,2048) | 1,050,624 |
| dense_1 | (None,1), sigmoid | 2,049 |
| dense_3 | (None,4), linear | 8,196 |

These are architecture specifications, not empirical recognition results. Runtime assertions
check **16,826,181 total and trainable parameters, 0 non-trainable** (float32 storage
64.19 MiB, displayed as MB by Keras). Summaries are printed before compilation to avoid
including optimizer state. `artifacts/verification/model_summary.txt` contains the actual
ImageNet-initialized summary from this environment.

The class head means **face present vs no face**, not identity or criminality and not mask/no-mask.
The bbox head predicts normalized xyxy coordinates for **one face**. Training images must
contain at most one face; no-face negatives have `class:0`, `subject:null`, `bbox:null`.
Positive images have `class:1`, a consented `subject`, `condition:"mask"` or `"no_mask"`,
a pixel bbox and a `session` identifier. See `data/training.example.json`.

Split policy: build connected components of volunteer identities and capture sessions,
then apply a single seeded `GroupShuffleSplit`, requested **75% train / 25% validation by
group**. Actual image proportions can differ. No identity or session can cross this split.
Each partition must contain masked positives, unmasked positives and no-face negatives.
Collect multiple independent groups, ideally each with both conditions and background images;
if the fixed split lacks coverage, collect additional groups rather than searching seeds.
Actual indices, identities, sessions and hashes are written to `localizer/split.json`.
No test set is used in fitting; reserve separate capture sessions for final evaluation.

Training only: OpenCV random crop (80–100% dimensions; accept a positive crop only if it
retains >=80% of bbox area), 50% horizontal flip, rotation uniformly within ±15 degrees
with reflected borders. Boxes undergo the same transforms, rotated corners produce an
axis-aligned enclosing box clipped to the image, then normalized after resizing to 120x120.
Use RGB followed by `keras.applications.vgg16.preprocess_input` (ImageNet BGR mean subtraction).
Validation is resized/preprocessed without augmentation.

Loss = binary cross-entropy + positive-only mean squared localization error, equal weights.
For bbox targets, the fifth training-target column is the presence mask, not a model output;
negative examples contribute zero bbox loss. Default Adam learning rate 1e-5, batch 8,
10 epochs, seed 2024. Python/NumPy/OpenCV/TensorFlow are seeded, TensorFlow deterministic
ops enabled, and CPU threads limited. A seeded RNG stream drives train shuffling and
augmentation. Reproducibility requires the same software, data, ordering and hardware;
FPS and floating-point results may vary across machines.

Outputs: `model.keras`, `model_summary.txt`, `split.json`, `training.json` with actual loss
history/configuration/versions. No trained model is shipped without volunteer training.
Reload with `from src.vgg_localizer import localization_loss` before
`tf.keras.models.load_model(...)` so the custom loss is registered.

Fig. 1's relationship to the Section IV dlib identity encoder is not specified by the paper.
They are implemented as **separate experiments**. The localizer does not replace or train
the 128-dimensional recognizer; no claim is made that localizer training improves recognition.

## Measurement definitions and outputs

`detectCriminal(frame, recognizer, ...)` returns an annotated copy and predictions.
Every camera frame is resized to about 0.25x, converted BGR->RGB, located, encoded and
matched to the nearest enrolled 128-d vector by Euclidean distance. Both
`compare_faces(..., tolerance)` and **`1 - distance > confidence_threshold`** must accept.
Ties choose the first enrollment entry in sorted filename order. Confidence is an
uncalibrated distance transform (can be negative), **not** a probability or accuracy estimate.
Green boxes and filled green uppercase name labels follow Figs. 8–9; rejected faces say UNKNOWN.
There is no mask-specific model, unmasking, temporal smoothing or automatic threshold tuning.

Ground truth is associated to predictions one-to-one using IoU >=0.5 (configurable).
The assignment maximizes admissible match count first and total IoU second. Each matched
pair yields an identity classification event; unseen consenting subjects have ground-truth
`__UNKNOWN__`. Each missed face yields `(identity, __MISSED__)`, each extra detection
`(__BACKGROUND__, predicted_identity)`. A truly empty correctly rejected frame contributes
one `(__BACKGROUND__, __BACKGROUND__)` event. Thus detection failures cannot disappear
from identity metrics. Background-only frames can raise accuracy; report their counts
from the per-class supports and keep condition composition comparable. Successful unknown
rejection still requires detecting the unknown face; missing it is a miss.

Accuracy is correct events / all scored events. Precision/recall/F1 use sklearn, macro
averaging over observed semantic classes (including unknown/background when observed;
excluding the artificial MISSED prediction class), `zero_division=0`. Per-class support
and confusion matrix are also saved. Macro F1 is the average per-class F1, not the harmonic
mean of macro precision/recall. Curves show **cumulative event accuracy separately for each
condition**, a definition chosen because the paper does not specify its curve aggregation.
Per-frame accuracy and raw event labels are retained as well. These metrics differ from
binary match/no-match rates and should not be claimed as directly comparable to unspecified
paper metrics. Aggregate results include both mask conditions in original manifest order.

Timing uses `perf_counter` around resize + detection + encoding + matching + drawing.
It excludes frame decoding, ground truth scoring, disk writes, display/waitKey, and plotting;
this is **processing FPS**, not end-to-end webcam throughput. No warmup frames are discarded.
Instantaneous FPS = 1 / elapsed seconds; aggregate FPS = frames / sum(elapsed).
**Efficiency = FPS × 100**, a scaled rate, not a percentage, utilization or independent metric.
The paper's efficiency formula is unknown and its graph is not exactly 100x its speed graph.

Each real evaluation writes strict `results.json` with all per-frame observations,
per-condition and overall summaries, input hashes, thresholds, seed and machine/library
versions; `frames.csv`; and four PNG plots:

| File | Title | Axes | Color/window |
|---|---|---|---|
| accuracy.png | Face Recognition Accuracy Over Frames | Frame Index / Recognition Accuracy (%) | blue, x=0–40 |
| fps.png | Face Recognition Speed Over Frames | Frame Index / Speed (fps) | green, x=2–25 |
| efficiency.png | Face Recognition Efficiency Over Frames | Frame Index / Efficiency | red, x=2–25 |
| processing_seconds.png | Face Recognition Processing Time Over Frames | Frame Index / Processing Time (s) | purple, x=2–25; additional plot |

Solid lines are mask, dashed lines no_mask. Speed/efficiency y ranges are automatic so
hardware results cannot be hidden by the paper's ranges. Accuracy uses 0–100%.
The first detected face frame of each condition yields `recognition_mask.png` and
`recognition_no_mask.png`, even if all decisions are UNKNOWN. Missing detections yield
no screenshot and an explicit missing-screenshot record; accepted matches are never forged.
Every output directory must be empty to prevent stale figures masquerading as current results.
README tables below are updated from the generated JSON summaries by the evaluation command.

<!-- measured:start -->
| Condition | Frames | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | FPS range | Efficiency range |
|---|---:|---:|---:|---:|---:|---|---|
| mask | — | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| no_mask | — | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

| Quantity | Paper shows (approx.) | My measured |
|---|---|---|
| Accuracy | ~80% plateau | unavailable |
| FPS | ~1.1–3.3 | unavailable |
| Efficiency | ~110–290; formula unspecified | unavailable |
<!-- measured:end -->

No measured difference can be explained before collecting data. In a real run, differences
can arise from CPU/GPU, dlib detector/backend, image resolution, library builds, timing
scope, enrollment size, capture quality, mask occlusion, identity composition and thresholds.
One enrollment image is particularly limiting for masks. The paper supplies approximate
curves rather than a reproducible annotated benchmark, so these are contextual comparisons,
not a statistical replication or a target to tune toward. Record hardware and sample counts
with any interpretation; report failed recognitions as well as successes.

## Ethics, bias, false positives and privacy

Only informed, consenting volunteers; no scraping, real criminal records, or reuse of
paper photos. Include consent for stored face templates, evaluation screenshots, retention
and deletion. Exclude unconsenting bystanders. Anyone may withdraw; delete their originals,
encodings, caches, models potentially trained on them, logs and screenshots as appropriate,
then rebuild affected outputs. Keep a consent record outside this repository.

Face embeddings are sensitive biometrics, not anonymization. Encodings are local NPZ
without pickle, written mode 0600; training/evaluation directories are mode 0700 and JSON
reports private. Source images and parent-directory permissions remain the operator's
responsibility; local storage is **not encryption**. Use encrypted disks, appropriate access
controls, short retention, no synchronized folders, and do not commit/share artifacts.
Screenshots and predicted identities are also sensitive. No remote gallery, telemetry,
real-person search, attendance database or law-enforcement integration is included.

Detection/recognition can have demographic biases, mask-related false negatives and false
positives even at conservative thresholds. Small volunteer samples cannot establish fairness
or operational safety. Where separately consented and adequately sampled, audit relevant
subgroups, lighting/pose/mask conditions and unknown-person false acceptance on a held-out
validation protocol; do not infer demographics from faces. A displayed name is a tentative
algorithmic match, not evidence of identity, wrongdoing or criminal status. Do not use this
demo for accusations, surveillance, access denial or other consequential decisions.

## Verification performed in the supplied environment

`artifacts/verification/pytest.xml` and `pytest.txt` contain the actual test run. All **35 tests
passed**, including real dlib distance matching and HOG/Haar blank-frame processing,
skipped enrollment errors, private NPZ round-trip, bbox scoring and metric arithmetic,
plot writing, CLI evaluation on synthetic empty images, Fig. 1 forward/gradient computation,
one optimizer step, custom-loss save/reload and identity/session split checks. Synthetic
fixtures are temporary software tests, never research data or the reported experiment.
The ImageNet-initialized summary was run separately. Third-party deprecation warnings
and sklearn single-label warnings remain in the log; they did not cause failures.

## Deviations and resolved ambiguities

1. **Data**: consenting volunteers only; no criminal records, random-person scraping or paper-photo reuse. Dataset is intentionally empty until consented inputs exist.
2. **Unavailable experiments**: no volunteer training, recognition benchmark, live webcam or Figs. 8–9 screenshots can be produced from the PDF alone; null results replace fabricated numbers. Empty-data reporting is an added mode.
3. **Section IV packaging**: the paper's described “simple faces” helper is organized into typed Python modules, CLI commands, tests and a Makefile; glob/os filename loading is retained.
4. **Multiple enrollment faces**: reject/skip ambiguous images instead of silently choosing somebody; duplicate enrollment identities and missing consent fail explicitly.
5. **Recognition backend**: HOG is the explicit default; CNN and Haar are selectable. Haar changes localization only. Exact quarter-size scaling is corrected for rounded frame dimensions.
6. **Threshold semantics**: confidence is exactly 1−Euclidean distance; nearest neighbor must pass both tolerance (inclusive comparison by face_recognition) and strict confidence threshold. Defaults 0.6/0.4 are declared choices, not optimized paper values. Unknown is explicit.
7. **Display**: green boxes/filled uppercase labels are preserved; rejected faces show UNKNOWN and the application title says Volunteer Recognition. `detectCriminal` does no attendance persistence.
8. **Fig. 1 vs recognizer**: no undocumented integration is invented. The independent trainable VGG16 localizer and pretrained dlib recognizer are separate; no PyTorch/MobileNet experiments are added from the introduction's mentions.
9. **Head meanings**: unspecified binary class is resolved as face presence; bbox is normalized xyxy for one face. This is not a binary criminal classifier or mask classifier. Positive data includes real masks and no masks.
10. **Layer names and memory**: names/shapes/parameter counts are matched to Fig. 1 explicitly. Pooling “512” denotes its output dimension; memory is float32 MiB labelled MB by Keras. The offline architecture test uses random weights; the production default and recorded summary use ImageNet.
11. **Loss/optimization**: unspecified localization loss is positive-only MSE; BCE+MSE have equal weights. Adam 1e-5, batch 8, 10 epochs and seed 2024 are documented defaults.
12. **Augmentation**: unspecified magnitudes/order are crop, flip, ±15° rotation with synchronized boxes, reflected borders, clipping and ImageNet preprocessing. Validation is unaugmented. Rotated boxes are enclosing rectangles.
13. **Split**: fixed seeded connected identity/session groups, 75/25 requested group split; save exact actual split. Independent test captures, unknown volunteers and background negatives are added to avoid leakage and expose false positives. Automated hashes do not prove independence.
14. **Accuracy definition**: joint detection/identity events, IoU=0.5 association, unknown/background/miss handling, macro sklearn metrics with zero_division=0 and cumulative within-condition accuracy are explicit choices; paper aggregation/averaging and ground truth are absent.
15. **Timing**: explicit processing-only timer, no warmup discarded, per-frame and aggregate FPS. Paper timing scope/hardware are insufficiently specified. No artificial delays or rate caps match the plots.
16. **Efficiency**: FPS×100 is a declared scaled-rate definition; the paper does not define its formula. Its approximate efficiency and FPS ranges need not be mutually consistent.
17. **Plots/screenshots**: exact figure titles/axis labels/colors, chosen per-condition line styles, required frame windows, automatic speed/efficiency y scales and an extra processing-time plot. No padded frames, recreated target curves or success-selected screenshots. Short/failed runs remain visibly incomplete.
18. **Additional Section V work**: user surveys, satisfaction, regulatory certification, vulnerability audit, multi-device compatibility study, public-space deployment/uptime and iterative field testing are not performed or claimed. No public-space deployment is included.
19. **Privacy/reproducibility**: consent attestations, local non-pickle encodings, permissions, provenance hashes, raw observations, deterministic settings and pinned dependencies extend the paper's underspecified implementation. They are not a legal compliance or fairness certification.
