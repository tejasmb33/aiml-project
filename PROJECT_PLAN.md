# Volunteer Face Recognition: Project Plan

## Purpose

Reproduce “OpenCV Based Enhanced Criminal Identification Mechanism” (IEEE SPARC 2024),
then compare its recognition pipeline with a proposed modern face recognition pipeline.
The application recognizes enrolled, consenting volunteers. It does not determine whether
a person is a criminal or infer wrongdoing from appearance.

This document records the agreed direction and next implementation steps. It is not a
claim that the proposed extension has already been implemented or measured.

## Current status

| Component | Status |
|---|---|
| Paper recognition pipeline: OpenCV + face_recognition/dlib | Implemented |
| Optional Haar face detector | Implemented |
| Fig. 1: trainable VGG16 with class and bounding-box heads | Implemented; parameter count verified |
| Training and evaluation commands | Implemented |
| Automated software validation | 35 tests passed in the recorded run; see artifacts/verification/ |
| Training on volunteer images | Pending consented data |
| Mask/no-mask recognition measurements and screenshots | Pending consented data |
| SCRFD + ArcFace recognition pipeline | Proposed; not yet implemented |
| MacBook installation and camera access | Not verified |

Existing research results are explicitly unavailable in
`artifacts/evaluation-unavailable/results.json`. Software test fixtures are not research
observations. The original `project.zip` and its Base64 export are preserved; this newly
created plan is not present in that original archive.

## Recommended architecture

```text
Local webcam frame or annotated image
                 |
          SCRFD face detector
                 |
      Facial landmarks and alignment
                 |
       ArcFace-trained embedding model
                 |
       Normalize embedding to unit length
                 |
 Compare with locally enrolled volunteer embeddings
                 |
    Best cosine similarity exceeds a threshold?
            /                       \
          Yes                        No
   Volunteer identifier             UNKNOWN
                 |
     OpenCV boxes and labels; local evaluation
```

Use SCRFD as the initial detector. RetinaFace is an alternative detector for a separate
future comparison, rather than an additional required stage. InsightFace is a toolkit
that can provide detection, alignment and recognition models. OpenCV handles frames and
drawing; it is a library rather than the identity model.

ArcFace is the training loss/method behind the selected recognition model. Start with a
pretrained embedding model; volunteer enrollment stores reference embeddings and does
not require training that network from scratch. Select a specific detector/recognizer
bundle before implementation and record exact model files, source, licence, hashes,
input preprocessing, alignment template and embedding size. Do not silently substitute
models when a download fails. InsightFace code and pretrained model weights may have
different licence terms; confirm the chosen weights permit the intended research/use.

Qwen-VL through Ollama is optional for future image explanations or conversational
features. It is not part of the proposed identity-matching pipeline or its benchmark.

## Keep the experiments separate

1. **Paper recognition baseline:** existing dlib embeddings and Euclidean-distance
   matching, with the original one-image-per-person enrollment convention.
2. **Proposed recognition extension:** SCRFD + aligned ArcFace embeddings and cosine
   similarity, evaluated on the same held-out captures.
3. **Paper Fig. 1 localizer:** the separate VGG16 face-presence and bounding-box model.
   Its relationship to the identity encoder is unspecified by the paper. Do not claim
   that training this localizer trains or improves either recognition encoder.

Preserve baseline commands and dependencies. Add the modern pipeline behind its own
command/backend option and record its runtime independently. A 512-dimensional embedding
is common for ArcFace models, but the selected model's actual output must determine
validation; it must not be forced to match the baseline's 128-dimensional embeddings.

## Volunteer data protocol

Obtain informed consent for capture, local biometric templates, screenshots, evaluation,
retention and deletion. Use pseudonyms; exclude unconsenting bystanders. Do not scrape
images, use real criminal records, or reuse the paper's sample photographs.

Keep the original `dataset/` convention for baseline enrollment. For an optional
multiple-image gallery, use a separate directory such as:

```text
data/
  enrollment_arcface/
    volunteer_id/
      front.jpg
      side.jpg
      mask.jpg
  validation/
  test/
```

These filenames are examples, not supplied data. Collect varied poses, lighting and real
mask/no-mask captures. Maintain annotations and consent manifests with image paths,
volunteer identifiers, capture-session identifiers, mask conditions and every visible
face's pixel bounding box.

Use three capture roles:

- **Enrollment:** build the reference gallery.
- **Validation:** choose thresholds and quality rules before examining test results.
- **Test:** final measurement with the settings frozen.

Known volunteers can appear in all roles, because the task is recognizing enrolled
people, but captures and sessions must be separate. Include separately consented,
unenrolled people to measure unknown-person rejection; reserve independent unknown
people for final testing. Never put crops, augmented copies or neighboring captures from
one session into different roles and call them independent tests.

For the existing VGG16 training experiment, retain the documented identity/session-group
train/validation split and reserve independent final test data.

## Matching and mask handling

Normalize each ArcFace embedding. Compare a query to the local gallery using cosine
similarity. With multiple gallery images, begin with the maximum per-image similarity
and a fixed nearest-identity decision; record this aggregation rule. Reject invalid or
zero-norm embeddings. Select the acceptance threshold using validation data only.

Record detector confidence separately from identity similarity. Neither value is an
identity probability. Do not reuse dlib's numerical tolerance or `1 - distance` threshold
for ArcFace: the embeddings and score definitions differ.

Return UNKNOWN below the acceptance threshold. A missing face is a detection failure,
not successful unknown rejection. Masked images can still cause failures. Enrollment
with masks is an experimental factor, not a guarantee of mask robustness. For a useful
comparison, first run both systems with the same one-image gallery, then measure the
effect of multiple enrollment images separately.

## Implementation tasks

- [ ] Confirm MacBook CPU architecture, macOS/Python versions, memory and camera access.
- [ ] Select and document a specific SCRFD + ArcFace model bundle and its licence.
- [ ] Add a separately pinned modern-model runtime, initially ONNX Runtime CPU.
- [ ] Add model-file hash verification and explicit initialization errors.
- [ ] Implement detection, landmarks, alignment and documented input preprocessing.
- [ ] Implement finite/unit-normalized embeddings and local gallery storage.
- [ ] Implement cosine matching, configurable threshold and UNKNOWN decisions.
- [ ] Add baseline-compatible prediction boxes and volunteer labels.
- [ ] Connect the backend to the existing annotated-frame evaluator.
- [ ] Add meaningful tests for alignment/preprocessing, normalization, matcher edge cases,
       gallery serialization and scoring compatibility.
- [ ] Validate the full pipeline on real consented inputs.
- [ ] Freeze validation-selected settings and run the held-out comparison.

These tasks are pending; proposed modules could be named `src/arcface_recognizer.py` and
`src/enroll_arcface.py`. Keep dependency changes isolated until compatibility is tested.
The existing Linux lockfile is not proof that the environment installs unchanged on
Apple Silicon. Validate Mac-compatible versions before producing a Mac lockfile.

## Evaluation protocol

Run both recognition systems on the same annotated mask/no-mask test frames and record:

- Detection misses and extra detections using the same IoU association policy.
- Identity accuracy, precision, recall and F1, separately by condition.
- Unknown-person false acceptance and known-person rejection/misidentification.
- Raw numerator/denominator counts for each rate; do not report zero when a denominator
  is absent. Distinguish rejection, wrong known identity and detection failure.
- Per-frame processing time, instantaneous FPS and aggregate throughput.
- Hardware, software/model versions, preprocessing, gallery size and fixed thresholds.

Keep the existing paper-style plots and raw JSON/CSV records. Report the timing scope
explicitly and use comparable frame resolutions. If end-to-end latency is measured,
include capture/display overhead and label it separately from inference processing FPS.
The existing efficiency convention, FPS multiplied by 100, is a scaled rate rather than
an independent performance measure or a percentage.

Report detector failures as well as recognition decisions. A small volunteer sample
cannot establish fairness. Do not infer demographic attributes from faces; any subgroup
study needs separate consent, adequate sample sizes and a documented analysis.

## Completion criteria

The extension is ready for comparison when its model provenance is fixed, its full
pipeline runs locally, meaningful tests pass, and complete annotated consented test data
is available. Claims of improvement require measurements on that held-out data.
Do not fabricate results, copy the paper's curves, or tune settings to its approximate
accuracy/FPS values. Store encodings, images, predictions and screenshots locally with
restricted access and a deletion policy.

## Opening in VS Code

This workspace is a cloud Linux environment, not the MacBook's local filesystem.
The current session has no available VS Code launcher (`code`) or desktop-control tool,
so opening the editor remotely has not been verified.

Once the updated project folder is transferred to the MacBook:

1. In VS Code, select **File > Open Workspace from File...**.
2. Select `project.code-workspace` in the project folder.
3. Open `PROJECT_PLAN.md` from the Explorer.
4. Use **Shift+Command+V** for the Markdown preview.

If the VS Code shell command is installed, the equivalent command from the local
project directory is:

```bash
code project.code-workspace PROJECT_PLAN.md
```

The cloud path `/workspace/project` will not exist on the MacBook unless the user creates
it; use the actual location of the transferred project. The workspace file does not
install extensions, packages or models automatically.
