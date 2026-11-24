# Repository snapshot

This repository contains the project source, tests, setup files, research documentation,
and historical dataset audit scripts. It excludes participant photographs, face encodings,
consent records, downloaded dataset images/archives, generated measurements/contact sheets,
virtual environments, and pretrained model binaries. These remain in the local working project.

Links in research audit reports may reference excluded local review artifacts. Historical
audit scripts depend on those local artifacts and are retained for reproducibility; their
presence does not authorize downloading data, starting training, or publishing participant data.

Current local enrollment uses a NumPy NPZ archive named `artifacts/volunteer_encodings.json`.
It is intentionally excluded. New users must create their own consented enrollment inputs.
The original encoding command defaults to `artifacts/encodings.npz`.

YuNet scripts expect `models/face_detection_yunet.onnx`; obtain appropriate weights and
preserve their source/license separately. The mask classifier also requires trained weights.
No model accuracy is implied by the presence of these source modules.

The original README records the initial research baseline. `PROJECT_PROGRESS.md` and the
newer analysis modules describe later work. The newer plotting/analysis scripts also import
Pandas, which is not currently listed in the original requirements files. The inspected local
environment had NumPy/TensorFlow/SciPy version conflicts; this upload does not resolve them.
