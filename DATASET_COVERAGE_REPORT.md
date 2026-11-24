# Mask dataset coverage report and proposed preparation plan

Date: 15 November 2026. Status: documentation inspection; image audit pending.

## Scope and evidence

The local `dataset/` contains only `.gitkeep`: MASK = 0, NO_MASK = 0.
`models/` contains only its README. The existing working files were not changed.
No dataset archives were downloaded, no images generated, and no training started.

Public dataset pages, original repositories, and authors' papers were inspected.
The counts below are publisher-reported, not independently verified counts of
usable images. A source photograph may contain multiple faces and both labels;
face-instance counts must not be described as independent photograph counts.
Unknown means not established by the sources reviewed, not necessarily absent.

## Candidate counts and licensing

| Dataset | Reported size and class counts | License / access | Decision |
|---|---|---|---|
| Humans in the Loop Medical Mask Dataset (MMD) | 6,024 photographs. An external original research comparison reports 6,758 masked, 2,085 unmasked, and 224 incorrectly masked face annotations. These are not locally audited totals. | Publisher states CC0 1.0 and public-domain image sourcing; current page requests a form for access. | Preferred core candidate; preserve publisher attribution despite CC0. |
| Face Mask Wearing Image Dataset v2 | 24,916 images across Correct/Incorrect wearing categories; binary class counts not established. No bare-face category documented. | CC BY 4.0, DOI 10.17632/8pn3hg99t4.2. | Supplement for real mask types; cap contribution by participant/session. |
| A Masked and Unmasked Face Dataset v2 | 330 masked + 330 unmasked face crops, 660 reported total. | CC BY 4.0, DOI 10.17632/xyc9h3wjxf.2. | Small balanced supplementary candidate, subject to provenance and image audit. |
| FMLD | 41,934 photographs; 29,532 correctly masked, 32,012 unmasked, 1,528 incorrectly masked face instances. | Repository MIT license; photographs inherited from MAFA/WIDER FACE. Photograph permissions require separate verification. | Hold out of approved combination until upstream terms are resolved. |
| MaskedFace-Net | 67,049 correctly masked + 66,734 incorrectly masked derived images in current README; no bare-face class supplied by this dataset. | CC BY-NC-SA 4.0 stated, plus original FFHQ per-image author/license metadata. | Exclude: digitally applied masks do not satisfy the proposed real-image collection policy. |
| PKU-Masked-Face | 10,301 face images, 1,018 identities; exact class totals not published on the reviewed page. | Academic-only, noncommercial; signed institutional agreement required. | Optional academic benchmark, not an unrestricted public core dataset. |

Sources: [MMD publisher](https://humansintheloop.org/resources/datasets/medical-mask-dataset/),
[MMD comparison and FMLD counts, Tables 1 and 3](https://www.mdpi.com/2076-3417/11/5/2070),
[wearing dataset v2](https://data.mendeley.com/datasets/8pn3hg99t4/2),
[masked/unmasked dataset v2](https://data.mendeley.com/datasets/xyc9h3wjxf/2),
[FMLD repository](https://github.com/borutb-fri/FMLD),
[MaskedFace-Net original repository](https://github.com/cabani/MaskedFace-Net),
[PKU publisher and access terms](https://www.pkuml.org/resources/pku-masked-face-dataset.html).

## Documented diversity and remaining gaps

| Dataset | Lighting | Expressions | Face angles | Other diversity |
|---|---|---|---|---|
| MMD | Bright/daylight, dim/dark, and shadow counts unknown for both classes. | No verified smiling/neutral/talking/laughing/serious breakdown. | Frontal/left/right/tilt counts unknown. | Publisher describes diversity of age, ethnicity, and region; accessory and other-covering annotations exist. Balance by skin tone, gender, face shape, glasses, beard, mask color, and indoor/outdoor setting unverified. |
| Wearing dataset v2 | Authors document natural and artificial illumination. This does not establish dark-room or shadow coverage, and does not establish bare-face coverage. | Requested expression categories unverified. | Authors report varied capture angles; directional/tilt counts unknown. | Bandana, cotton, N95, surgical and correct/incorrect placement documented. Child/Male/Female folders are documented; these mixed age/gender categories do not prove broad demographic balance. Other requested factors remain unverified. |
| Masked/unmasked v2 | Unknown. | Unknown. | Unknown. | Medical-mask and bare-face crops documented; demographic, accessory, environment and mask-style diversity unknown. |
| FMLD | Requested lighting strata unknown. | Requested categories unknown. | Pose annotations documented. | Annotated gender and ethnicity exist; neither establishes skin-tone balance. Real masks and placement labels documented. |
| PKU | Variation documented for masked and unmasked images; exact bright/dim/shadow strata unknown. | Requested categories unknown. | Front, left, right, up, down documented; roll/tilt coverage unknown. | Different mask types documented; other requested factors unknown. |

The wearing dataset details are from the [authors' data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10700517/).
There is currently no verified candidate combination covering all nine requested
requirements, especially MASK and NO_MASK under real dim lighting and shadows.
No source reviewed establishes all five requested expressions in both classes.
Talking and laughing should not be asserted from an ambiguous still image, and
mouth expressions concealed by a mask must remain unknown unless reliable source
annotations or video context establish them.

## Formats, corruption, and duplicates

Local dataset: zero images, so there are no image formats to inventory or images
to decode/hash. For every external candidate, format distribution, corrupt count,
exact duplicate count, near-duplicate count, and post-filter class totals are
**not yet measured**. No clean-data claim is made.

FMLD explicitly reuses MAFA and WIDER FACE, so those collections must not be
counted as independent additions. Cross-source overlap among other candidates
also requires checking. Large image counts do not establish many unique people.

## Proposed combination

Start with audited MMD face annotations as the core; add a participant-balanced
subset of the wearing dataset for mask-type variation, and the small CC BY
masked/unmasked dataset only if its image/provenance audit passes. These are
candidates, not a finalized or sufficient training set. Do not sum their reported
counts into a purported usable total.

Prioritize additional licensed real NO_MASK and MASK data for any measured
lighting, expression, pose, or demographic gaps. If suitable public data cannot
be found, report the gaps; separately consented webcam collection can be proposed
for approval. Synthetic images and mask overlays are excluded. No photographs
are to be scraped and no access forms submitted on the user's behalf in this step.

## Preparation procedure proposed for approval

1. Acquire only accepted sources and retain immutable originals. Preserve each
   release README, license text, DOI/version, author attribution, citations,
   retrieval date, source URLs, archive SHA-256, and any per-image credit. Track
   each crop back to its original image and record transformations. Keep license
   boundaries explicit; a repository code license does not replace image terms.
2. Fully decode images to verify integrity; inventory actual format, dimensions,
   orientation and channels. Log invalid files without deleting originals. Check
   annotation bounds, incorrect labels, face size, blur, and unusable exposures.
3. Detect exact duplicates using file and normalized-pixel hashes. Generate
   perceptual-hash candidates for manual near-duplicate review across all sources.
   Check conflicting labels and repeated photos/crops before splitting.
4. Keep two classes: MASK = a physical face mask worn over at least the mouth,
   including a nose-exposed mask, with wearing style retained as metadata;
   NO_MASK = no physical mask worn over the mouth, including chin-only masks.
   This is a proposed presence policy, not a compliance classifier. Review all
   improper-wearing examples individually. A scarf/hand/face shield alone is
   NO_MASK only when absence of a physical mask can be established; ambiguous
   occlusions are excluded and logged, not forced into a class.
5. Store lighting, expression, pose, mask type/color/style, accessories, and
   environment as metadata. Use unknown/not-visible where evidence is absent.
   Keep original demographic annotations with provenance; do not infer gender,
   ethnicity or precise age from appearance. Do not substitute ethnicity for
   skin tone or brightness for lighting-condition ground truth.
6. Produce per-class coverage cross-tabs, including class x lighting x pose,
   and expressions where identifiable. Use face-crop luminance only to flag
   possible lighting cases for manual review. Record audited counts, unknown
   counts, unique-person/session counts where known, and exclusions.
7. Group source photographs, duplicate families, known identities, and capture
   sessions before a proposed 70/15/15 train/validation/test split. Respect
   published test partitions. Document residual identity leakage risk where IDs
   are unavailable. Include a source-held-out evaluation where feasible. Never
   place augmented versions of the same original in different splits.
8. Crop each annotated face using the same bounding-box convention as runtime,
   retain the full nose/mouth/chin region, convert to RGB, resize to 224x224,
   then apply MobileNetV2 preprocess_input exactly once to obtain [-1, 1].
   Match the existing direct-resize behavior. Avoid background-only samples as
   a third class. Do not discard masked faces merely because HOG misses them.
9. After approval, consider modest training-only exposure/contrast/color changes,
   blur/noise, flips, and small rotations/crop jitter, retaining labels and
   identifying transformed images as augmentation. These transformations are
   not evidence of real low-light coverage. If no image transformations are
   wanted, omit augmentation. Keep validation/test images unaugmented except
   deterministic inference preprocessing.
10. Balance classes and available diversity strata without letting one subject
    or source dominate. Keep an evaluation slice for each verified lighting
    condition in each class. A missing slice blocks claims of robustness there.

## Architecture and approval boundary

The current classifier requires a full `.keras`/`.h5` model containing a named
MobileNetV2 backbone, input `(None, 224, 224, 3)`, and two softmax outputs. Keep
manifest class names MASK and NO_MASK, with output mapping 0=MASK, 1=NO_MASK;
the existing UI renders the second label as `NO MASK`. Do not embed another
input normalization stage. A later training implementation can use MobileNetV2,
global average pooling, dropout, and a two-unit softmax head.

Evaluate classifier performance by condition and end-to-end webcam detection
separately: the existing HOG stage can miss faces in masks, darkness, or profile
before classification occurs. Do not change that working stage during preparation.

Approval requested for the candidate acquisition/audit and preparation plan only.
Training requires explicit approval. Exact accepted class totals and a completed
image-level coverage report must precede any assertion that the dataset is ready.
