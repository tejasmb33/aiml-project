# HITL staging audit — excluded at provenance gate

15 November 2026. **Partial image audit, not a completed dataset audit. Training prohibited.**

## Decision

Exclude this release from the training pool under the user's requirement to exclude
sources with unverifiable image provenance. The publisher and declared CC0 license
were verified, but the release provides no original-photo URLs, original authors,
or per-image rights evidence. All 6,024 annotation JSON records and the class
metadata were checked for provenance fields; none were present. A public-domain
sourcing statement alone does not independently establish each photo's provenance.
This is an evidence gap, not a finding that HITL violated anyone's rights.

Acquisition stopped once that gap was established. Retained staging data comprises
all annotation files, the complete archive directory index, and 1,195 fully received
images. The other 4,822 images were not acquired or decoded. No files were deleted.
The partial archive and range fragments are evidence, not a usable full ZIP release.
No complete-archive SHA-256/MD5 validation is claimed; hashes of received transfers
and CRC checks on recovered ZIP entries are recorded separately.

No existing project file was modified: SHA-256 verification checked 36 pre-existing
files and found zero changed or missing. MobileNetV2, preprocessing, output mapping,
and existing webcam modules remain unchanged. Neither CC BY supplement was acquired
or evaluated further, and no training or dataset split was started.

## Original source, attribution, and license

- Publisher: Humans in the Loop; annotations credited to its refugee workforce in
  Bulgaria using Supervise.ly.
- [Original dataset page](https://humansintheloop.org/resources/datasets/medical-mask-dataset/).
- [Publisher's Kaggle release](https://www.kaggle.com/datasets/humansintheloop/medical-mask-detection),
  version 1, released 30 March 2021, owner `humansintheloop`.
- Declared license: **CC0 1.0 Universal** on the original page; Kaggle also declares
  CC0/Public Domain. [CC0 legal code](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en).
- CC0 permits copying, adaptation and redistribution, including commercial use under
  copyright. It does not clear third-party privacy/publicity rights or verify the
  rights of the person applying CC0. The unresolved issue here is photograph
  provenance, not a missing publisher license declaration.
- The historical S3 endpoint returned HTTP 404. The publisher's Kaggle API supplied
  the release. The current HITL webpage's linked license sentence also points to
  an unrelated daily-objects page; this link was not used as archive provenance.

Preserved evidence: `evidence/hitl_source.html`, `evidence/kaggle_metadata.json`,
`evidence/CC0-1.0-legalcode.txt`, `evidence/provenance_annotation_scan.json`,
`evidence/archive_directory_index.json`, `evidence/received_extraction.json`, and
`evidence/received_transfer_hashes.json`. Publisher attribution is retained even
though CC0 does not impose an attribution condition.

## Actual counts and release inconsistencies

The complete ZIP directory index contains **6,017 image entries**, **6,024 annotation
entries**, and one `meta.json`. These are measured release-entry counts, not the
publisher's stated 6,024-image count. An earlier progress message estimated 6,020;
the exact parsed count is 6,017.

Seven numeric image stems each have two annotation files with different original
extensions: 6100, 6101, 6107, 6108, 6137, 6166, 6200. The image payload names use
`.jpg`. Converting a name's extension therefore does not always give an unambiguous
annotation match. Those seven cases require inspection, not arbitrary selection.

| Source face label | Records across all 6,024 JSON files | Records attached to 1,195 decoded images | Decoded photographs containing this label |
|---|---:|---:|---:|
| face_with_mask | 6,769 | 1,685 | 885 |
| face_no_mask | 2,086 | 345 | 243 |
| face_with_mask_incorrect | 227 | 56 | 47 |
| face_other_covering | 1,786 | 272 | 208 |

These are source-label counts, before semantic corrections and box exclusions.
They are not validated training counts. A photograph may contain both classes.
The ambiguous stems inflate some counts: excluding all seven stems leaves 2,062
unmasked and 1,778 other-covering records; correctly/incorrectly masked counts
remain unchanged. See `audit/full_annotation_inventory.json`.

**Accepted training examples: MASK = 0; NO_MASK = 0.** Incorrect wearing and other
coverings were not silently reassigned. The future class names stay MASK and
NO_MASK; lighting, pose, and expression remain metadata.

## Image-level integrity and duplicates — received subset only

- Fully decoded and verified: **1,195 images**, all actually JPEG.
- Corrupt images found in that subset: **0**. No conclusion about unreceived images.
- Exact file duplicates: **0 groups**; normalized decoded-pixel duplicates: **0 groups**.
- Perceptual duplicate candidates: **0 pairs** at 64-bit dHash Hamming distance <=4.
  This narrow screen does not rule out other near duplicates, repeated identities,
  transformed crops, or train/test leakage.
- Invalid annotation boxes: **14 across 11 photographs**: 11 masked, 2 incorrectly
  masked, 1 unmasked. Detailed coordinates appear in `audit/invalid_boxes.json`.
  Several are zero-area boxes; e.g. 0202.jpg has [250,21,250,21].
- Annotation JSON parsing errors: **0**. Every decoded image has a matching numeric
  annotation stem, and none of the seven ambiguous stems is in this received subset.

The decoded images are an archive-prefix convenience subset, not a random or
statistically representative sample of the complete dataset. No population-wide
coverage percentages are estimated from it.

## Visually observed coverage

Twenty-four face records received explicit metadata review after contact-sheet
inspection. This small review demonstrates examples, not exhaustive coverage.
Metadata counts in `audit/reviewed_coverage_counts.json` are grouped by **original
source label**, including the apparent wrong-label case; they are not class truth.

| Factor | Observed examples | What remains unverified |
|---|---|---|
| Daylight / well-lit | Masked 1072.jpg:0 outdoors and 1112.jpg:0 indoors; unmasked 0437.jpg:1 outdoors and 0391.jpg:0 indoors. | Full per-class counts and measured illumination levels. |
| Dim / night / shadows | Masked 0121.jpg:2 and 1148.jpg:0 in low-lit interiors; 0031.jpg:11 and 0852.jpg:0 at night. Unmasked 1003.jpg:10 has outdoor facial shadow; 0634.jpg:0 is a shadowed low-key portrait. | A dark background does not establish a dark room. Verified real dark-room NO_MASK coverage remains missing. |
| Face angle / pose | Frontal faces in both labels; masked left/right profiles in 0852.jpg:0 and 1112.jpg:0; unmasked side view in 0178.jpg:4 and in-plane tilt in 1171.jpg:0. | Balanced left/right/tilt coverage for each class and lighting combination. |
| Visible expressions | Unmasked visible smiles in 0391.jpg:0 and 1171.jpg:0; open-mouth smile in 0437.jpg:1; neutral-appearing faces in 0447.jpg:0 and 0353.jpg:7. | Talking versus laughing cannot be established from these stills. Serious expressions and hidden masked-mouth expressions are unverified. |
| Masks / accessories | Surgical-style, respirator-style, black and patterned masks; spectacles, sunglasses, hats, hair coverings, facial hair and occlusion occur in inspected samples. | Exhaustive type/color/style counts and combinations with low lighting. |
| Environments | Indoor clinical/transit/restaurant-like scenes, outdoor streets/gardens, and nighttime scenes. | Coverage balance and performance in each environment. |
| Demographics / face shape | No demographic attributes inferred from faces during this audit. | Quantified age, skin-tone, gender, and face-shape coverage. Publisher diversity claims are not an audited distribution. |

The reviewed shadowed NO_MASK portrait is not evidence of a dark-room acquisition.
The apparent wrong-label dim example is also not evidence of correct masked-class
coverage. Lighting and expressions are never additional training classes.

## Label and quality findings

Visual checks do not certify every annotation. Source annotations remain unchanged.

- **0038.jpg:13** is labelled `face_with_mask`, but the crop shows an uncovered nose
  and mouth. Logged as an apparent incorrect mask label; proposed binary label
  NO_MASK requires subsequent review before use.
- **0696.jpg:0** is labelled improperly masked, with an uncovered nose and mouth.
  Under the proposed mouth-coverage presence policy it would map to NO_MASK. This
  is a mapping review, not proof that the source's improper-wearing label is wrong.
- **1011.jpg:2** has its lower face hidden behind an airplane seat. This does not
  establish a physical mask, so no false-positive label-error claim is made; its
  binary class cannot be visually verified from the crop.
- **0064.jpg:3 and :5** are very low-quality reflection/background-like regions.
  They cannot support confident face/mask verification and need quality exclusion
  or manual review. They are not counted as proven wrong labels.

See `audit/issues_reviewed.json` and `audit/label_error_examples.jpg`. The entire
release has not been manually checked, and an overall label-error rate is unknown.

## Representative samples and machine-readable outputs

[Open the selected coverage gallery](audit/representative_samples.jpg). Each cell
shows the full image context at left and the annotated face crop at right, with an
image/annotation ID. Crops/resizing are for inspection, not generated training data.

![Selected observed coverage and label-review examples](audit/representative_samples.jpg)

Additional sheets include `audit/face_with_mask_darkest_1.jpg`,
`audit/face_no_mask_darkest_1.jpg`, and the brightest/spread sheets. Brightness
ranking merely selects review candidates; it does not provide lighting ground truth.

- `audit/images.json`: dimensions, actual formats, hashes, luminance screening values.
- `audit/faces.json`: subset face records; all excluded from training eligibility.
- `audit/manual_review.json`: explicit observations for 24 faces, with unknowns retained.
- `audit/full_annotation_manifest.json`: original records with exclusion reasons.
- `audit/annotation_filename_collisions.json`: seven ambiguous filename groups.
- `audit/summary.json`: partial-audit scope and counts.
- `evidence/existing_files_verification.json`: preservation verification.

## Next step

HITL is excluded pending original-photo provenance evidence from the publisher
(e.g. a release manifest linking photographs to original URLs/authors and applicable
rights evidence). No forms were submitted and nobody was contacted on the user's
behalf. A mirrored copy without additional provenance would not resolve this gap.

The two CC BY datasets remain deferred; this first-source inspection does not
justify adding them automatically. Their acquisition would be a separate step
under the same provenance condition. No training is authorized by this audit.
