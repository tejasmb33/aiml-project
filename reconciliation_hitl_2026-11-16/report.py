from pathlib import Path
from collections import Counter
import json,hashlib,re
from PIL import Image,ImageOps,ImageDraw
out=Path(__file__).resolve().parent;r=Path('staging_hitl_2026-11-15');raw=r/'extracted'
summary=json.loads((out/'planning_summary.json').read_text());counts=json.loads((out/'reconciliation_counts.json').read_text())
faces=json.loads((r/'audit/faces.json').read_text());by_id={f['id']:f for f in faces}
# Explicit reviewed examples rather than fabricated coverage counts.
examples=[('0444.jpg:1','Bright daylight / MASK'),('0444.jpg:5','Bright daylight / NO_MASK'),('0065.jpg:0','Normal/mixed indoor / MASK'),('0065.jpg:3','Normal/mixed indoor / NO_MASK'),('0045.jpg:4','Dim interior / MASK'),('0045.jpg:0','Dim interior / NO_MASK'),('0611.jpg:0','Night street / MASK'),('1080.jpg:0','Darker scene / NO_MASK; time unknown')]
sheet=Image.new('RGB',(1800,1800),'white');draw=ImageDraw.Draw(sheet)
example_metadata=[]
for i,(id,caption) in enumerate(examples):
 f=by_id[id];x,y=(i%4)*450,(i//4)*900
 with Image.open(raw/f['image']) as im:
  rgb=im.convert('RGB');full=ImageOps.contain(rgb,(440,390));crop=ImageOps.contain(rgb.crop(tuple(f['bbox'])),(440,440))
 sheet.paste(full,(x,y+50));sheet.paste(crop,(x,y+455));draw.text((x+3,y+3),id,fill='black');draw.text((x+3,y+22),caption,fill='black')
 example_metadata.append(dict(face_id=id,source_image=f['image'],source_class=f['source_class'],visual_scene_assessment=caption,luma_mean_screening_value=f.get('luma_mean'),note='Visual example, not calibrated lighting measurement or a population coverage count'))
sheet.save(out/'DIVERSITY_LIGHTING_SAMPLES.jpg',quality=94)
(out/'lighting_example_metadata.json').write_text(json.dumps(example_metadata,indent=2))
# A manifest schema for future preparation; no directories/crops or labels applied.
schema=dict(class_indices={'MASK':0,'NO_MASK':1},runtime_display_labels=['MASK','NO MASK'],input_shape=[None,224,224,3],input_color='RGB',normalization='tensorflow.keras.applications.mobilenet_v2.preprocess_input exactly once, externally',output_shape=[None,2],output_activation='softmax',originals_immutable=True,proposal_only=True,training_allowed=False)
(out/'proposed_dataset_contract.json').write_text(json.dumps(schema,indent=2))
structure='''prepared_mask_v1/                         # future derived directory; not created
  README.md                               # scope, limitations, source attribution
  metadata/
    class_indices.json                    # MASK=0, NO_MASK=1
    sources.json                          # original URLs, authors, licenses, rights evidence
    images.csv                            # 1,195-image disposition ledger
    faces.csv                             # original labels, approved labels, bbox, review status
    diversity.csv                         # lighting, visible expression, pose; unknown retained
    exclusions.csv                        # audit holds; no deletion of originals
    invalid_boxes_original.json           # 14 original points, kept separately
    split_groups.csv                      # image/duplicate-family/session/person groups
    transformations.json                  # crop/resize/RGB/preprocessing version
  splits/
    train.csv                             # unassigned until approval and grouping checks
    validation.csv
    test.csv
  crops/                                  # future derivatives, never original replacements
    train/
      MASK/
      NO_MASK/
    validation/
      MASK/
      NO_MASK/
    test/
      MASK/
      NO_MASK/
'''
(out/'PROPOSED_DIRECTORY_STRUCTURE.txt').write_text(structure)
selection=json.loads((out/'balanced_selection_PROPOSAL.json').read_text())
selected_coverage={label:dict(Counter(v['lighting'] for v in selection if v['proposed_label']==label)) for label in ['MASK','NO_MASK']}
(out/'proposed_metadata_coverage.json').write_text(json.dumps(selected_coverage,indent=2))
report='''# Dataset reconciliation report — HITL

Date: 16 November 2026. Based on DATASET_AUDIT_V2.md and the existing staged files.
Scope: reconciliation and preparation planning only. No internet requests, new
acquisition, original annotation edits, original image changes, or training.

## Outcome

All **1,195 decoded photographs** are accounted for. All **120 mixed-annotation
photographs** were reviewed using original context with class boxes and enlarged
representative crops. Their coexistence usually reflects different people, not a
contradictory label on one face. Five important exception scenes are documented
below. Additional low-quality and mouth-visibility issues remain on review hold.

A **conditional balanced selection proposal** references **181 MASK and 181 NO_MASK
face crops**, from **342 source photographs**. Twenty photographs contribute one
candidate to each class. No actual training crop directory or split was created.
The proposed records remain unapproved and not eligible for training. The existing
original-photo provenance exclusion remains unresolved; accepted training examples
are still zero. The selected 362 records are a reviewable planning subset, not a
claim of a clean or robust training dataset.

## 1. All 120 mixed-label photographs

Review all photographs in the [contact-sheet viewer](reconciliation_hitl_2026-11-16/CONTACT_SHEET.html).
It contains 12 full-resolution pages, original-image links, and per-image findings.
Green rectangles show original MASK annotations; red rectangles show original
NO_MASK annotations. Annotation indices are zero-based. These drawings are review
derivatives, not replacements for original images or corrected ground truth.

[Compact overview of all 120](reconciliation_hitl_2026-11-16/CONTACT_SHEET_ALL_120.jpg).
The overview is for navigation; use the full-resolution pages for face detail.

| Finding at scene/context level | Photographs |
|---|---:|
| Multi-person scene with separate MASK/NO_MASK annotation regions, without one of the five exception patterns below | 115 |
| Single unmasked portrait with an invalid MASK point | 1 |
| NO_MASK regions are printed advertisement faces | 1 |
| NO_MASK region is a tablet-screen face | 1 |
| NO_MASK region is a cardboard-cutout face | 1 |
| NO_MASK rectangle mainly shows flowers/hair/occluded region | 1 |
| Total reviewed | **120** |

This is a scene-level assessment, not certification of every individual face label.
Among the 115 multi-person scenes, small background faces, rear heads, reflections,
and blur can prevent mask-state verification. Source labels must not be treated as
correct solely because rectangles belong to different people.

### Five exception scenes

| Image | Observation | Proposed handling; nothing applied |
|---|---|---|
| 0291.jpg | A single unmasked portrait. The MASK record is a zero-area point; a valid NO_MASK box already exists. | Keep invalid record separately. Do not count this as a usable mixed-class scene. |
| 0641.jpg | Two physically present masked people; NO_MASK records 6 and 7 annotate faces printed on an advertisement. | Hold depicted-face crops from the live-person training proposal; retain original labels. |
| 0775.jpg | A physically present masked person and a NO_MASK face on a tablet. | Hold the screen-face crop for the live-person dataset; original image remains intact. |
| 0793.jpg | Two masked people and a life-size cardboard figure with an unmasked depicted face. | Hold the cardboard crop. |
| 1256.jpg | Two people are present, but NO_MASK record 1 mainly captures flowers/hair and an occluded area rather than an inspectable mouth. | Hold that crop; do not infer a replacement label or box. |

A printed or displayed face can truthfully depict an unmasked face. Those records
are a **domain/policy mismatch** for a live physically-present-person dataset, not
necessarily an error in the source's broader annotation policy.

The geometric screen found **zero valid MASK/NO_MASK rectangle pairs with IoU >=0.30**
in these 120 photographs. That argues against obvious overlapping opposite-class
boxes, but it does not prove semantic label correctness. The invalid point in
0291.jpg is outside that positive-area comparison.

Other recorded concerns include 0064.jpg (reflections/tiny regions), 0082.jpg and
0581.jpg (downward/rear heads), 0113.jpg and 1042.jpg (blurred background faces),
0120.jpg (physical mask versus beard ambiguity), and 1148.jpg/1197.jpg (small blurred
background faces). These examples are marked in the per-image review ledger.

Every image's assessment is in
[mixed_review.csv](reconciliation_hitl_2026-11-16/mixed_review.csv) and
[mixed_images_reviewed.json](reconciliation_hitl_2026-11-16/mixed_images_reviewed.json).
Contact-sheet pages:
'''
report+=' '.join(f'[{i}](reconciliation_hitl_2026-11-16/mixed_{i:02d}.jpg)' for i in range(1,13))+'\n'
report+='''
## 2. Reconciliation of all 1,195 decoded photographs

The reported image-class counts overlap:

- 885 images have at least one original MASK record.
- 243 images have at least one original NO_MASK record.
- 120 images are included in both counts.
- Union: **885 + 243 - 120 = 1,008** images.
- Remainder: **1,195 - 1,008 = 187** images.

A mutually exclusive ledger accounts for every photograph:

| Source-annotation presence | Unique photographs |
|---|---:|
| MASK present, NO_MASK absent | 765 |
| NO_MASK present, MASK absent | 123 |
| Both present | 120 |
| Neither main label present | 187 |
| Total | **1,195** |

These are annotation-presence counts, not fully verified semantic whole-image labels.
On valid geometry alone, the first three rows become 765, 124, and 119 because the
spurious MASK point in 0291.jpg no longer creates a mixed scene. The 187 remainder
is unchanged. Mixed photographs can supply separately labelled face crops; the
entire source photograph must never receive two conflicting binary image labels.

### What the 187 images contain in their source annotations

| Exact source-label subgroup | Unique photographs |
|---|---:|
| OTHER_COVERING face labels only, among the four source face-state categories | 145 |
| Incorrectly-worn face labels only | 18 |
| Both OTHER_COVERING and incorrectly worn; no main MASK/NO_MASK labels | 2 |
| No face-state annotation at all; accessory annotations only | 22 |
| Total | **187** |

“Only” here does not exclude hat/mask/accessory rectangles. It refers to source
face-state labels, not the actual absence of additional people or face conditions.
The 145 other-covering images must not automatically become NO_MASK, and the 18+2
improper-wearing images need the mouth-coverage mapping reviewed face by face.

The 22 accessory-only images have 14 surgical-mask, 13 colorful-mask, and one hat
annotation. Those 28 rectangles are not 28 annotated human faces. Visual inspection
shows loose masks, statues, mannequin/display heads, a toy, a cake, and a human-like
masked portrait without a face annotation. A missing face-state label is not evidence
of a bare face. Human authenticity and new face boxes would need separate review;
none was invented here.

[View all 22 accessory-only images](reconciliation_hitl_2026-11-16/no_face_state_22.jpg).
All 187 images have their own ledger entry and appear in eight additional sheets:
'''
report+=' '.join(f'[{i}](reconciliation_hitl_2026-11-16/neither_{i:02d}.jpg)' for i in range(1,9))+'\n'
report+='''
Complete accountability files:
[all 1,195 images](reconciliation_hitl_2026-11-16/image_ledger.csv),
[187 remainder images](reconciliation_hitl_2026-11-16/neither_images.json), and
[22 accessory-only records](reconciliation_hitl_2026-11-16/no_face_state_22.json).
No image was discarded, duplicated into the training pool, or assigned a class by default.

## 3. Re-review of the four unresolved improperly worn records

![All four cases, original context above enlarged crop](reconciliation_hitl_2026-11-16/unresolved_four.jpg)

| Original record | New visual assessment | Proposal, not approved |
|---|---|---|
| 0215.jpg:7 | In the oblique side view the mask appears lowered below the mouth; pose/arm and limited pixels constrain certainty. | Tentative NO_MASK, medium confidence; second review required. |
| 0800.jpg:8 | Small oblique profile appears to show the mask below the mouth. Upscaling cannot recover missing detail. | Tentative NO_MASK, medium confidence; second review required. |
| 1128.jpg:1 | Mouth appears above the top edge of a low mask. | Tentative NO_MASK, medium confidence; second review required. |
| 0767.jpg:0 | Black neck/hood covering could be scarf/balaclava rather than a separate physical face mask. | Remain unresolved pending the covering-definition policy and second review. |

The rule proposed in V2 remains: MASK if a recognizable physical face mask covers
the mouth; NO_MASK if the mouth is uncovered. Nose exposure is wearing-style metadata
under this presence policy, not an extra classifier class. Scarves/balaclavas need
an explicit scope decision. This policy does not certify correct mask use.

If the three tentative mappings were approved, the prior 56-record proposal would
become 31 MASK, 22 NO_MASK, one unresolved and two invalid. That is **not** an applied
relabeling. All four cases and all other improperly worn/other-covering records
remain outside the balanced selection until their mapping is approved.
[Structured four-case review](reconciliation_hitl_2026-11-16/unresolved_four_review.json).

## 4. The 14 invalid boxes remain separate

The 14 originals were neither expanded nor changed. They remain zero-area points.
Their source annotations and images are intact. Each apparent target already has
a valid counterpart; four have a class conflict with that counterpart. A copied
reference audit, not a replacement annotation file, is stored at
[invalid_boxes_separate_REFERENCE_ONLY.json](reconciliation_hitl_2026-11-16/invalid_boxes_separate_REFERENCE_ONLY.json).
See V2 for all coordinates and paired valid records. No guessed boxes or duplicate
training faces were produced. Future cleanup belongs in a derived approved manifest.

## 5. Proposed balanced dataset — selection plan only

The deployment model needs individual face crops. Balancing whole scenes would
mix faces and labels. The attached proposal selects **one candidate crop per class
per source photograph**, and balances at the crop level while also giving each
class the same number of contributing source photographs.

Proposed screening rules:

1. Start only from original main MASK and NO_MASK face records with valid geometry.
2. Hold the 14 invalid points, the unapproved wearing-style/other-covering records,
   and known semantic-quality concerns separately. No source records are changed.
3. Require a proposed minimum native crop short side of 64 pixels. This is a review
   heuristic to limit severe upscaling, not a guarantee of sharpness or a proven
   MobileNetV2 requirement. Resizing a tiny crop to 224 pixels does not add detail.
4. Keep the largest remaining face per class per photograph as the initial proposal.
   The largest face is not automatically the best; reviewers can propose substitutions.
5. Prioritize explicit prior visual observations, then use a stable hash ordering
   with the fixed prefix `reconciliation-v1:` to choose a reproducible MASK subset.
   No validation/test accuracy was used for selection.

| Planning stage | MASK | NO_MASK |
|---|---:|---:|
| Valid main-label face boxes before semantic/size review | 1,674 | 344 |
| Source photographs with a candidate after proposed holds/64-pixel screening and one-per-class-per-photo | 814 | 181 |
| Selected candidate crops / contributing source photographs per class | **181** | **181** |
| Approved training examples | **0** | **0** |

The 362 proposed crops reference **342 unique photographs**, with 20 shared between
classes. Each shared photograph must stay within one split. No cropped training
image, new class label, source-image copy, or train/validation/test partition was
created. Nonselected candidates remain available in the inventory; they were not deleted.

[Balanced selection proposal — CSV](reconciliation_hitl_2026-11-16/balanced_selection_PROPOSAL.csv)
and [JSON](reconciliation_hitl_2026-11-16/balanced_selection_PROPOSAL.json).
Each row includes source hash, original annotation label, proposed class, original
box, group ID, unassigned split, metadata, review status and `training_eligible=false`.
[Full face disposition inventory](reconciliation_hitl_2026-11-16/face_disposition_inventory.json).

This selection is balanced by count, **not yet balanced by lighting, pose, expression,
identity, or demographic appearance**. Many labels/quality attributes are unverified;
all records still fail the original-photo provenance gate. It is a reviewable small
prototype proposal, not an adequate demonstrated real-world training corpus.

### Later splitting and preparation, if approved

- Group all crops from the same source photograph, exact/near-duplicate family,
  known person and known capture session before splitting. Existing person/session
  identifiers are incomplete, so perfect identity independence cannot be promised.
- Propose approximately 70/15/15 by groups. For 181 examples per class, 127/27/27
  per class is an illustrative target only; no assignment is made. Grouping and
  coverage take precedence over exact arithmetic, and unresolved groups stay held.
- Keep training balanced after grouping; do not independently shuffle face crops
  or generate matching augmentations across partitions. Holdout distributions
  must be reported; 27 examples per class would be a small preliminary test only.
- Derive a face crop using approved original bounds, retain nose/mouth/chin, convert
  to RGB, resize directly to 224x224 to match the current runner, and apply MobileNetV2
  preprocess_input to [-1,1] **exactly once**. Store derivative images unnormalized;
  preprocessing occurs at the model input.
- Preserve output order MASK=0, NO_MASK=1, two-unit softmax. The current UI displays
  NO_MASK as `NO MASK`; no working module needs changing for the directory proposal.

## 6. Lighting diversity: supported versus unverified

The four requested conditions remain metadata. Scene context and illumination of
an individual face are different facts: a dark background can coexist with a lit
face. These are visual assessments, not lux measurements or calibrated exposure labels.

| Requested condition | Actual image evidence | Current limit |
|---|---|---|
| Bright daylight | 0444.jpg contains both classes in an outdoor sunlit park; 0883.jpg has a clear bare face outdoors. | Balance by class/pose/person and exact per-condition totals across all 1,195 remain unverified. |
| Normal indoor lighting | 0065.jpg shows a masked passenger and bare faces in an illuminated transit interior. 1118.jpg has both classes in an interior. | Lighting can be mixed with window daylight; per-face exposure needs annotation. |
| Dim indoor lighting | Re-review of 0045.jpg shows both MASK and NO_MASK faces in a relatively dim artificial-light interior. 0979.jpg and 1148.jpg have dim church-like scenes but bare background faces are often blurred. | This adds a clearer NO_MASK dim-scene example than the earlier audit, but does not establish a sufficiently large independent low-light group or very dark-room coverage. |
| Night / low light | 0611.jpg shows a clear masked person on a night street. 1080.jpg has a clear unmasked face with a darker artificially lit background; 0611.jpg's bare background face is blurred. | NO_MASK nighttime coverage is weaker; 1080.jpg does not establish exact time of day. Real dark-room/noisy low-exposure performance remains unverified. |

![Lighting examples: context above face crop](reconciliation_hitl_2026-11-16/DIVERSITY_LIGHTING_SAMPLES.jpg)

The earlier intensity screen found 259 MASK/50 NO_MASK bright-candidate photographs,
and 97 MASK/37 NO_MASK dark-candidate photographs (valid geometry, overlapping
source-class counts). These are screening counts, not actual-lighting labels.
For example, 0045.jpg's source NO_MASK crop means are about 67 and 76, so a <=60
threshold misses this dim-scene example. Conversely hair/black masks can darken
crops in bright daylight. No source was relabelled on intensity alone.

The balanced proposal includes reviewed examples plus many metadata-unknown records.
Its exact inherited metadata counts are in
[proposed_metadata_coverage.json](reconciliation_hitl_2026-11-16/proposed_metadata_coverage.json).
Unverified metadata is retained rather than filled with guesses. Neither class
currently has demonstrated sufficient coverage across all four conditions.

## 7. Facial diversity: actual support and gaps

| Requirement | Supported by inspected images | Still unverified |
|---|---|---|
| Smiling | Visible bare-face smiles in 0391.jpg:0, 1171.jpg:0 and 1080.jpg:0. | Expression balance in each class/lighting group; a mouth hidden by a mask cannot be labelled from appearance alone. |
| Serious | Non-smiling/neutral-looking bare faces occur, e.g. 0447.jpg:0 and 0353.jpg:7. | A deliberately serious expression or internal emotional state is not established by a still; no verified serious-expression distribution exists. |
| Talking | Interaction scenes and open-mouth stills exist. | A still does not reliably establish talking versus another action; verified talking labels require source/video context. No talking-coverage claim is made. |
| Head angles and poses | Frontal views, masked left/right profiles (0852.jpg:0 and 1112.jpg:0), bare side view (0178.jpg:4), downward views and visible in-plane tilt (1171.jpg:0). | Counts and balanced class x lighting x pose coverage; rear-head boxes are not evidence of a classifiable profile. |
| Different people and appearances | Numerous multi-person scenes; spectacles, sunglasses, facial hair, hair/face coverings and varied visible appearance occur. | Unique-person count, capture-session count, and balanced age/skin-tone/gender/face-shape representation. No sensitive demographics were assigned from appearance. |

The same subjects may recur across photographs; unique files are not unique people.
The report does not infer masked expressions or fabricate diversity annotations.
Every unverified factor remains unknown. Count balancing alone cannot establish
robustness or correct the existing HOG detector's possible misses in darkness,
profiles or masks.

## 8. Proposed clean directory structure

This is a future derived directory, **not created**. Original staged images and
annotations stay in their current locations, untouched, with hashes and references.

```text
'''+structure+'''```

Only MASK and NO_MASK are class directories. Lighting, pose, visible expression,
wearing style, accessories and environment belong in metadata files. An unresolved
record is an audit hold, never a third output class. Save attribution, license and
per-image provenance alongside source records before any admission to training.
The saved model must still contain the named MobileNetV2 backbone expected by the
existing loader, accept (None,224,224,3), and emit (None,2) softmax probabilities.

[Dataset contract proposal](reconciliation_hitl_2026-11-16/proposed_dataset_contract.json)
and [directory plan](reconciliation_hitl_2026-11-16/PROPOSED_DIRECTORY_STRUCTURE.txt).

## 9. Preservation and review boundary

All original images, source JSON, prior reports, archive fragments, and working
project files were hash-snapshotted before this phase and checked afterward.
[Preservation evidence](reconciliation_hitl_2026-11-16/preservation_check.json).
Only this new report and `reconciliation_hitl_2026-11-16/` were created. No additional
images were downloaded or recovered from the 364 stored fragment entries.
No source label, original pixel, bounding box, existing code, or prior report was changed.

Next review decisions are the mouth-coverage/covering-type policy, the three tentative
NO_MASK cases, the proposed quality/size screens and balanced selection, and the
missing provenance evidence. No training is authorized or started by this report.
'''
Path('DATASET_RECONCILIATION_REPORT.md').open('x').write(report)
# Verify all pre-existing bytes and the completeness of the new review artifacts.
before=json.loads((out/'preservation_before.json').read_text());changed=[]
for name,expected in before.items():
 p=Path(name)
 if not p.is_file():changed.append(name);continue
 h=hashlib.sha256()
 with p.open('rb') as stream:
  while b:=stream.read(1048576):h.update(b)
 if h.hexdigest()!=expected:changed.append(name)
checks=dict(preexisting_files_checked=len(before),changed_or_missing=changed,source_images_relabelled=0,original_boxes_modified=0,new_acquisition=False,training_started=False)
(out/'preservation_check.json').write_text(json.dumps(checks,indent=2))
assert not changed,changed
assert len(json.loads((out/'mixed_images_reviewed.json').read_text()))==120
assert sum(counts['partitions'].values())==1195 and sum(counts['neither_subgroups'].values())==187
assert len(selection)==362 and Counter(v['proposed_label'] for v in selection)=={'MASK':181,'NO_MASK':181}
assert all(not v['training_eligible'] and v['proposal_only'] for v in selection)
missing=[x for x in re.findall(r'\]\(([^)]+)\)',report) if not Path(x).exists()]
assert not missing,missing
print('Report written; all links resolve;',len(before),'pre-existing files unchanged.')
