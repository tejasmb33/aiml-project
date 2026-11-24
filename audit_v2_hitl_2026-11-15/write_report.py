from pathlib import Path
from collections import Counter
import json,hashlib
from PIL import Image,ImageOps,ImageDraw
out=Path(__file__).resolve().parent;r=Path('staging_hitl_2026-11-15')
balance=json.loads((out/'unique_image_balance.json').read_text())
faces=json.loads((r/'audit/faces.json').read_text());by_id={f['id']:f for f in faces}
chosen=[('1072.jpg:0','MASK: daylight'),('1112.jpg:0','MASK: well-lit interior'),('0611.jpg:0','MASK: night street'),('0979.jpg:13','MASK: dim interior'),('0391.jpg:0','NO_MASK: well-lit interior'),('0437.jpg:1','NO_MASK: daylight'),('1080.jpg:0','NO_MASK: dark background/artificial light'),('0634.jpg:0','NO_MASK: shadowed portrait; room unknown')]
sheet=Image.new('RGB',(1600,1100),'white');draw=ImageDraw.Draw(sheet)
for i,(id,caption) in enumerate(chosen):
 f=by_id[id];x,y=(i%4)*400,(i//4)*550
 with Image.open(r/'extracted'/f['image']) as im:
  rgb=im.convert('RGB');full=ImageOps.contain(rgb,(390,230));crop=ImageOps.contain(rgb.crop(tuple(f['bbox'])),(390,250))
 sheet.paste(full,(x,y+50));sheet.paste(crop,(x,y+290));draw.text((x+3,y+3),id,fill='black');draw.text((x+3,y+23),caption,fill='black')
sheet.save(out/'lighting_examples.jpg',quality=94)
proposals=json.loads((out/'improper_proposals.json').read_text());invalid=json.loads((out/'invalid_box_reviews.json').read_text())
report='''# DATASET AUDIT V2 — HITL Medical Mask Dataset

Date: 15 November 2026. Scope: existing local material only; no network requests,
new dataset acquisition, training, source-image changes, or source-label changes.

## Findings

- The 1,195-image audit limit was caused by deliberate termination of the download
  at the prior provenance gate, followed by extraction of only the primary download
  prefix. It was not caused by 4,822 corrupt images, an HTTP failure for each image,
  or a dataset split.
- My prior description of all 4,822 remaining images as “unreceived” was inaccurate.
  Of those entries, 364 have complete CRC-verified data in retained fragments;
  16 have partial data; 4,442 have no retained entry bytes.
- The 1,195 unique decoded photographs contain source MASK annotations in 885
  photographs and source NO_MASK annotations in 243. **120 photographs contain both**.
- All 56 improper-wearing records were visually reviewed. Proposals: 31 MASK,
  19 NO_MASK, 4 uncertain, and 2 zero-area records. Nothing was relabelled.
- All 14 invalid boxes were reviewed: each is a point, and every apparent target
  already has a valid annotation in its photograph. No point can be automatically
  expanded into a justified face box. Four have conflicting counterpart classes.
- Bright and darker-scene examples exist, but sufficient real low-light coverage
  in both classes is **not established**. Brightness screening is not illumination
  ground truth; the reviewed subset also contains annotation and visibility issues.

The original release remains excluded from training under the existing provenance
condition. That decision did not delete its staged images or annotations. Accepted
training counts remain zero; descriptive source-label counts below are separate.

## 1. Why only 1,195 images were decoded

The expected publisher ZIP size was **768,818,472 bytes**. The retained primary
transfer is **162,492,416 bytes**, about 21.14% of the ZIP. Percentage of bytes is
not percentage of images because compressed image sizes differ.

During the previous turn I started a sequential download, then parallel range
requests to finish it. After all 6,024 annotations were inspected and no original
photo URLs/authors/per-image rights metadata were found, I deliberately sent SIGTERM
to the download processes. Their exit status was 143 (termination by SIGTERM).
There was no completed-archive checksum validation. The earlier range probes also
had timeouts, and the legacy S3 link returned 404; neither explains the exact
1,195-image boundary.

`extract_received_prefix.py` reads the primary ZIP stream sequentially, checks each
complete entry's size and CRC, and stops at the first incomplete entry. It recovered
6,024 annotation entries plus 1,195 image entries. The last complete image was
`1270.jpg`; `1271.jpg` was incomplete. File numbers are not a continuous count.
The extractor did not assemble or inspect the separate range fragments.

The full central-directory index was recovered independently from tail ranges,
which is why 6,017 image names could be inventoried without receiving every image.
It contains 6,017 images, 6,024 annotation files and one class metadata file.
The seven extra annotation files are filename-stem collisions, not seven extra
image files. The publisher's webpage count of 6,024 images does not match this ZIP.

### Exact disposition of the other 4,822 image entries

| Disposition | Image entries | Meaning |
|---|---:|---|
| Complete entry data retained, not part of the 1,195 decoded images | 364 | Fragment reconstruction in memory now verifies each entry's uncompressed size and CRC; no new image was written or added to the image audit. |
| Some entry bytes retained, incomplete | 16 | At transfer boundaries; cannot yet reconstruct complete entries from local bytes. |
| No entry bytes retained | 4,442 | Transfers were terminated before these intervals arrived. |
| Total not in original decoded subset | **4,822** | 364 + 16 + 4,442. |

Thus 1,195 + 4,822 = 6,017. None of the latter categories is a finding of source
image corruption. CRC success for the 364 complete entries is byte integrity, not
an image-decoding or semantic audit. They are deliberately not included in the
1,195-image statistics requested here.

Forensic method: parse actual ZIP64 compressed sizes and local-record offsets;
union all retained primary/range/tail byte intervals; compare every remaining image
entry to those intervals. Per-image results and CRC evidence are available in
[remaining_image_status.json](audit_v2_hitl_2026-11-15/remaining_image_status.json),
[download_forensics.json](audit_v2_hitl_2026-11-15/download_forensics.json), and
[complete_fragment_entries_crc.json](audit_v2_hitl_2026-11-15/complete_fragment_entries_crc.json).
No transfer was resumed and no internet data was acquired for V2.

## 2. What the annotation counts mean

The model classifies individual **face crops**, not entire source photographs.
A source photograph may have many face annotations, including both classes and
additional accessories. A mask/accessory rectangle is not another independent
face training example. Different files also do not guarantee different people.

| Original face label | Face records in the 1,195-image subset | Valid-box face records | Unique photographs containing label | Unique photographs containing a valid box for label |
|---|---:|---:|---:|---:|
| face_with_mask (MASK) | 1,685 | 1,674 | 885 | 884 |
| face_no_mask (NO_MASK) | 345 | 344 | 243 | 243 |
| face_with_mask_incorrect | 56 | 54 | 47 | 45 |
| face_other_covering | 272 | 272 | 208 | 208 |

These are recomputed from the staged annotations associated with the actual decoded
files. Valid geometry does not certify the label: errors such as `0038.jpg:13`
remain possible. Annotation indices in this report are **zero-based**.

## 3. Unique-image class balance

### Separate counts requested

- **MASK: 885 unique photographs** with at least one original MASK annotation
  (884 if only valid face boxes are counted).
- **NO_MASK: 243 unique photographs** with at least one original NO_MASK annotation.

These totals overlap. They must not be summed as independent whole-image classes.

| Mutually exclusive source-annotation presence category | Unique images | Percentage of all 1,195 |
|---|---:|---:|
| MASK present, NO_MASK absent | 765 | 64.02% |
| NO_MASK present, MASK absent | 123 | 10.29% |
| Both MASK and NO_MASK present | 120 | 10.04% |
| Neither of these two main source labels | 187 | 15.65% |
| Total | **1,195** | **100.00%** |

“Absent” means absent from source annotations, not visually proven absent from the
photograph. Other-covering and improper-wearing annotations may coexist with any
row. The 187 photographs must not automatically become NO_MASK examples.

Only-valid-box analysis gives 765 MASK-without-NO_MASK, 124 NO_MASK-without-MASK,
119 mixed images, and the same 187 neither. The difference is `0291.jpg`: a zero-area
MASK point overlaps an existing valid NO_MASK portrait annotation. This is a concrete
reason raw annotation presence can misstate actual class balance.

MASK appears in approximately 3.64 times as many photographs as NO_MASK under the
raw labels. This indicates an imbalance, but is not a mutually exclusive training
class ratio. The valid face-crop counts are 1,674 versus 344 before semantic review,
a separate imbalance measure. Do not label a mixed photograph with one whole-image
class or allow crops from one photograph to cross dataset splits.

A fully verified semantic class balance would require checking all relevant face
annotations. This V2 review verifies all 56 improper cases and all 14 invalid points,
plus the lighting-review samples; it does not claim an exhaustive label-accuracy
review of every face in all 1,195 photographs.

Machine-readable counts: [unique_image_balance.json](audit_v2_hitl_2026-11-15/unique_image_balance.json).

## 4. All 56 improperly worn mask annotations

There are 56 face records across 47 photographs, including 54 positive-area crops
and two zero-area points. Every record was reviewed with the full photograph and
its face crop where one exists. Full review sheets are linked in Appendix A.

### Proposed two-class policy, awaiting approval

- MASK: a recognizable physical face mask covers the mouth. A nose-exposed mask
  remains MASK under this **presence** policy; its wearing style is metadata.
- NO_MASK: no mask covers the mouth, including a mask on the chin, hanging from an
  ear, on the forehead, or covering only the eyes.
- Ambiguous coverage/type or invalid geometry: keep the original record untouched
  and mark it unresolved in audit metadata. This is not a third classifier class.

This policy does not test correct mask use. If nose-and-mouth compliance is intended,
the nose-exposed cases need a different approved mapping. No such policy change
has been made to the existing two-output model.

| Proposed disposition | Face records |
|---|---:|
| MASK | 31 |
| NO_MASK | 19 |
| Needs second review | 4 |
| Invalid geometry; no usable crop from that record | 2 |
| Total | **56** |

Four uncertain cases are `0215.jpg:7`, `0767.jpg:0`, `0800.jpg:8`, and `1128.jpg:1`.
They involve a small/oblique face, uncertain mouth coverage, or uncertain covering
type. The zero-area cases are `0295.jpg:0` and `0784.jpg:0`; their existing valid
counterparts are respectively MASK and OTHER_COVERING. They must not be expanded
or mapped into new duplicate crops automatically.

Representative cases:

| Example | Observation | Proposed mapping |
|---|---|---|
| 0012.jpg:2 | Mask under chin; mouth exposed | NO_MASK |
| 0271.jpg:1 | Mouth covered, nose exposed | MASK |
| 0613.jpg:2 | Mask on forehead | NO_MASK |
| 0711.jpg:0 | Mask over eyes; mouth exposed | NO_MASK |
| 0737.jpg:0 | Several masks cover the face and mouth | MASK |
| 1053.jpg:0 | Printed teeth are a mask design | MASK |
| 1219.jpg:1 | Mask covers mouth during a nasal swab | MASK |
| 0767.jpg:0 | Black scarf/balaclava versus physical mask unclear | Unresolved |

![Representative proposals, with original context and crops](audit_v2_hitl_2026-11-15/improper_representative_proposals.jpg)

These are review proposals, not source relabellings or final accepted training
labels. See [improper_proposals.json](audit_v2_hitl_2026-11-15/improper_proposals.json)
for every observation, original label, geometry status and proposal.

## 5. All 14 invalid bounding boxes

Every invalid rectangle has x1=x2 and y1=y2. Its width and height are both zero.
They are not merely out-of-bounds boxes that clipping would repair. A single point
does not preserve the missing face dimensions; adding an arbitrary margin would
invent geometry and potentially include another face.

Visual inspection and sibling-annotation checks found a valid box for each apparent
target. Therefore the likely remediation is to retain the existing valid annotation
and propose removal/ignoring of the redundant point in a **future derived manifest**,
subject to approval. No original annotation is to be deleted, changed or overwritten.
Duplicating the valid rectangle into the point's record would double-count that face.

| # | Invalid record | Original zero-area box | Existing valid record in same image | Existing valid box | Assessment |
|---|---|---|---|---|---|
'''
for v in invalid:
 id=f"{Path(v['image']).name}:{v['index']}"
 mate=f"{Path(v['image']).name}:{v['existing_valid_counterpart_index']}"
 label=v['existing_valid_counterpart_label']
 text='Class conflict; retain valid counterpart pending review' if v['class_conflict'] else 'Redundant point; valid face already exists'
 report+=f"| {v['review_number']} | {id} ({v['label']}) | {v['bbox']} | {mate} ({label}) | {v['existing_valid_counterpart_bbox']} | {text} |\n"
report+='''
None of the 14 has a justified automatic point-to-rectangle correction. All have
an existing valid counterpart, so this is largely annotation cleanup rather than
recovery of missing face images. In `0264.jpg`, three separate zero-area records
(indices 3, 4 and 5) refer to the apparent target already represented by valid
record 6; another point refers to valid record 1.

The four counterpart class conflicts are:

- `0291.jpg:0`: MASK point versus an existing NO_MASK portrait; do not create a MASK crop.
- `0295.jpg:0`: improper-mask point versus an existing valid MASK face; that still
  shows mouth coverage and does not need a new face rectangle.
- `0784.jpg:0`: improper-mask point versus OTHER_COVERING with scarf/bandana metadata;
  the physical-covering policy still requires review.
- `1044.jpg:4`: MASK point versus OTHER_COVERING for the apparent target; a covering
  type/label conflict needs review, not geometric guessing.

Reviewed sheets with the original point marked by a red cross:
[1–4](audit_v2_hitl_2026-11-15/invalid_boxes_01.jpg),
[5–8](audit_v2_hitl_2026-11-15/invalid_boxes_02.jpg),
[9–12](audit_v2_hitl_2026-11-15/invalid_boxes_03.jpg),
[13–14](audit_v2_hitl_2026-11-15/invalid_boxes_04.jpg).
[Structured box assessments](audit_v2_hitl_2026-11-15/invalid_box_reviews.json).

## 6. Bright- and dark-light sufficiency

**Sufficient coverage in both classes is not established.** Presence of a few
examples is not evidence of enough distinct people, sessions, poses, expressions,
and environments for real-world robustness. There is no agreed deployment target,
minimum subgroup size, or held-out lighting benchmark to demonstrate sufficiency.

### Screening all relevant valid face crops

For a reproducible screening pass only, mean grayscale crop intensity <=60 on an
8-bit scale flags a dark candidate; >=130 flags a bright candidate. These are
explicit audit heuristics, not calibrated lighting definitions. Skin reflectance,
hair, black masks, pose, blur and framing all affect this statistic.

| Source class | Unique photographs with valid face boxes | Bright candidate photographs | Dark candidate photographs | Photographs with both candidate types |
|---|---:|---:|---:|---:|
| MASK | 884 | 259 | 97 | 10 |
| NO_MASK | 243 | 50 | 37 | 0 |

Underlying face counts are 344 bright and 123 dark MASK candidates, and 58 bright
and 40 dark NO_MASK candidates. The photograph columns count each image once per
class/threshold, even if several faces qualify. They do not replace actual lighting
labels, and faces between the thresholds are not automatically a third condition.

All 40 dark NO_MASK candidates (37 photographs) were visually inspected, along with
24 spread-selected dark MASK candidates and earlier bright/daylight examples.
Review showed why intensity cannot establish illumination: many low-intensity
crops are from daytime outdoor scenes or well-lit transit scenes, include mostly
hair, have blur/occlusion, or show a portrait with a dark background. For example,
`1256.jpg:1` is a flower/occluded region rather than a reliably visible face crop;
it is not evidence of useful NO_MASK low-light coverage. No label or box was changed.

### Actual visual observations and limits

- **Bright/daylight:** both classes have clear examples: MASK `1072.jpg:0` outdoors
  and `1112.jpg:0` in a bright interior; NO_MASK `0437.jpg:1` outdoors and `0391.jpg:0`
  in a well-lit interior.
- **MASK in darker contexts:** `0611.jpg:0` is on a nighttime street; `0979.jpg:13`
  and `1148.jpg:0` are in warm, relatively dim church-like interiors. The prior audit's
  restaurant-like description of `1148.jpg` was imprecise; the full context shows
  pews and a church-like interior. `1197.jpg` is the restaurant-like scene.
- **NO_MASK in darker contexts:** `1080.jpg:0` is a clearly unmasked face with a
  dark/artificially lit background. This provides a darker-scene example, not proof
  of a dark-room acquisition or a measured low exposure at the face. `0611.jpg:2`
  is a small blurred background face in the nighttime street; it is weaker evidence
  and should not be treated as a validated high-quality low-light sample.
- **Shadows:** NO_MASK `1003.jpg:10` has facial shadow outdoors; `0634.jpg:0` is a
  low-key portrait whose room lighting cannot be inferred from its black background.
  MASK daytime-shadow examples also exist, e.g. `0338.jpg:9`.
- **Dark rooms:** a sufficiently large, verified NO_MASK dark-room group remains
  unestablished. Exact actual-lighting totals across all 1,195 images are still
  unknown: this audit did not manually assign a lighting label to every image.

There is evidence of both labels across brighter and darker contexts, but fewer
NO_MASK source images, uneven quality, and no verified balanced lighting cross-tab.
I cannot conclude that either lighting stratum has enough independent data for
robust training, particularly NO_MASK in real dim interiors. No augmentation was
used to manufacture coverage and no image was generated.

![Selected lighting examples; full context and face crops](audit_v2_hitl_2026-11-15/lighting_examples.jpg)

[Screening counts and candidate IDs](audit_v2_hitl_2026-11-15/lighting_proxy_screen.json).
Dark NO_MASK review sheets:
[1](audit_v2_hitl_2026-11-15/dark_screen_face_no_mask_01.jpg),
[2](audit_v2_hitl_2026-11-15/dark_screen_face_no_mask_02.jpg),
[3](audit_v2_hitl_2026-11-15/dark_screen_face_no_mask_03.jpg),
[4](audit_v2_hitl_2026-11-15/dark_screen_face_no_mask_04.jpg).

## 7. Preservation, model compatibility and remaining decisions

Only this new report and the new `audit_v2_hitl_2026-11-15/` artifacts were created.
The staged original images, source annotations, prior reports, fragments, code,
architecture and preprocessing were preserved. All 7,338 pre-existing project and
staging files captured before V2 were hash-checked after report creation; see
[preservation_check.json](audit_v2_hitl_2026-11-15/preservation_check.json).

The existing architecture remains MobileNetV2 with RGB 224x224 inputs,
preprocess_input exactly once, and two softmax outputs. The MASK/NO_MASK names
remain the two classification labels. Lighting, pose and expression remain metadata.
No training, acquisition, relabelling, cleanup or original-file deletion occurred.

Pending decisions, with no action taken:

1. Approve the physical-mouth-coverage mapping before any derived labels are made.
2. Resolve the four ambiguous improper-wearing crops and four invalid-point class
   conflicts before deriving a cleaned manifest.
3. If permitted later, reconstruct the 364 locally complete image entries and audit
   them separately; this does not require a download. Do not treat them as already
   decoded or included in the present balance and lighting counts.
4. Resolve the original provenance condition before admitting this release to training.
5. Establish and review a real lighting cross-tab by class, unique image, person and
   session before claiming dataset sufficiency. No additional dataset was acquired.

## Appendix A. Every improper-wearing record and proposal

Every source record remains unchanged. “Unresolved” and “invalid geometry” are audit
statuses, not extra training classes. The proposal does not certify correct wearing.

| Review # | Image:annotation | Proposed binary label / status | Observation | Review sheet |
|---|---|---|---|---|
'''
for f in proposals:
 label=f['proposed_binary_label'] or f['audit_status']
 report+=f"| {f['review_number']} | {f['id']} | {label} | {f['visual_observation']} | [view](audit_v2_hitl_2026-11-15/{f['sheet']}) |\n"
Path('DATASET_AUDIT_V2.md').open('x').write(report)
# Preserve all prior project and staging files, not merely the working source modules.
before=json.loads((out/'preservation_before.json').read_text());changed=[]
for name,expected in before.items():
 p=Path(name)
 if not p.is_file():changed.append(name);continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 if h.hexdigest()!=expected:changed.append(name)
(out/'preservation_check.json').write_text(json.dumps(dict(preexisting_files_checked=len(before),changed_or_missing=changed),indent=2))
assert not changed,changed
print('Report created. Preservation verified for',len(before),'pre-existing files.')
