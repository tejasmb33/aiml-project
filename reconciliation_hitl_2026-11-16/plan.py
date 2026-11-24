from pathlib import Path
from collections import Counter,defaultdict
import json,csv,hashlib
from PIL import Image,ImageOps,ImageDraw
out=Path(__file__).resolve().parent;r=Path('staging_hitl_2026-11-15');raw=r/'extracted'
faces=json.loads((r/'audit/faces.json').read_text());images=json.loads((r/'audit/images.json').read_text());by_image=defaultdict(list)
for f in faces:by_image[f['image']].append(f)
mixed=json.loads((out/'mixed_images.json').read_text())
special={
 '0291.jpg':('single_person_spurious_point','Single unmasked portrait; MASK exists only as an invalid zero-area point. Preserve and separate that point.'),
 '0641.jpg':('printed_face_annotations','Two physically present masked people; the NO_MASK rectangles annotate printed advertisement faces.'),
 '0775.jpg':('screen_face_annotation','A physically present masked person and NO_MASK face on a tablet display; not a second physically present unmasked person.'),
 '0793.jpg':('cardboard_face_annotation','Two masked people and a life-size cardboard cutout carrying the NO_MASK annotation.'),
 '1256.jpg':('occluded_or_nonface_annotation','Two people present, but the NO_MASK rectangle mainly captures flowers/hair and an occluded region; mask state cannot be verified.')}
notes={
 '0026.jpg':'Small blurry background NO_MASK face: class verification remains weak.',
 '0037.jpg':'Foreground MASK and distant blurry NO_MASK crop; distinct people but weak background quality.',
 '0064.jpg':'Includes tiny reflection-like face annotations; some faces cannot be confidently classified.',
 '0082.jpg':'NO_MASK crop is a downward/back-of-head view; mouth visibility is inadequate.',
 '0113.jpg':'Clinical foreground MASK is clear; NO_MASK background is very blurred.',
 '0117.jpg':'Side/rear head crops require mouth-visibility checks.',
 '0120.jpg':'The selected MASK crop has beard/low resolution; physical mask versus facial hair needs individual review.',
 '0233.jpg':'Many overlapping regions of different people; some background NO_MASK labels are blurry.',
 '0264.jpg':'Contains four invalid MASK points in addition to valid physical face records; keep invalids separate.',
 '0314.jpg':'NO_MASK is mostly side/back-of-head; class needs individual verification.',
 '0383.jpg':'Background NO_MASK region is blurry; distinct person but label weakly verifiable.',
 '0581.jpg':'NO_MASK crop is a back-of-head/ear view; do not infer mouth state.',
 '0601.jpg':'Background NO_MASK small/blurred; twilight/night cannot be asserted from color alone.',
 '0611.jpg':'Night street with clear foreground MASK and blurry background NO_MASK.',
 '0679.jpg':'NO_MASK region mostly hair/head profile; visible mouth evidence limited.',
 '0749.jpg':'At least two physical people, but selected NO_MASK is a distant low-detail head.',
 '0804.jpg':'Foreground lowered masks also appear; source labels do not exhaust all wearing styles.',
 '0937.jpg':'Several annotated MASK regions show side/back heads with weak mouth evidence.',
 '0979.jpg':'Dim church-like interior; background NO_MASK faces are small and blurred.',
 '1035.jpg':'Foreground unboxed bare face visible as well as annotated background; source boxes are not exhaustive.',
 '1042.jpg':'Background NO_MASK annotations heavily blurred.',
 '1085.jpg':'NO_MASK crop mostly back/side of head; cannot certify mouth state.',
 '1148.jpg':'Dim church-like interior; background NO_MASK labels not reliably readable.',
 '1183.jpg':'NO_MASK rear-head crop does not establish absence of a mask.',
 '1197.jpg':'Restaurant-like interior; NO_MASK background crop is blurred.'}
for v in mixed:
 name=v['filename'];kind,note=special.get(name,('multi_person_separate_regions','Multiple physical people with separate MASK/NO_MASK annotation regions. No obvious same-face contradiction at contact-sheet scale; individual labels still require review.'))
 v.update(visual_review='reviewed_context_and_representative_crops',scene_assessment=kind,review_note=note+(' '+notes[name] if name in notes else ''),same_face_cross_class_iou_ge_0_3=False,source_annotations_modified=False)
(out/'mixed_images_reviewed.json').write_text(json.dumps(mixed,indent=2))
with (out/'mixed_review.csv').open('w',newline='') as stream:
 keys=['review_number','filename','scene_assessment','valid_mask_count','valid_no_mask_count','review_note'];w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows([{k:v[k] for k in keys} for v in mixed])
# New observations, never edits to prior decisions or source annotations.
unresolved=json.loads((out/'unresolved_four_source.json').read_text())
updates={
 '0215.jpg:7':('NO_MASK','medium','Side-view mouth appears exposed with mask lowered below it; arm/pose limit certainty. Proposed only; needs approval and a second reviewer.'),
 '0800.jpg:8':('NO_MASK','medium','Oblique profile appears to have mask lowered below the mouth. Low pixel detail prevents a definitive final label.'),
 '1128.jpg:1':('NO_MASK','medium','Mouth appears above the top edge of the low mask; second review still required.'),
 '0767.jpg:0':(None,'unresolved','Black neck/hood covering may be balaclava/scarf rather than a separate face mask. Keep on hold until covering policy and image interpretation are agreed.')}
for v in unresolved:
 label,confidence,note=updates[v['id']]
 v.update(reconciliation_proposed_label=label,proposal_confidence=confidence,reconciliation_observation=note,approved=False,training_eligible=False)
(out/'unresolved_four_review.json').write_text(json.dumps(unresolved,indent=2))
# Preserve invalid originals separately as reference-only copies of audit records.
invalid=json.loads(Path('audit_v2_hitl_2026-11-15/invalid_box_reviews.json').read_text())
(out/'invalid_boxes_separate_REFERENCE_ONLY.json').write_text(json.dumps(invalid,indent=2))
# A conditional balanced selection proposal: one crop per class per photograph,
# >=64 pixels on shortest side. No crop files or dataset splits are created.
known_holds={f['id']:'NO_MASK region is printed, on a display/cardboard cutout, or occluded/nonface' for f in faces if f['source_class']=='face_no_mask' and Path(f['image']).name in {'0641.jpg','0775.jpg','0793.jpg','1256.jpg'}}
known_holds['0038.jpg:13']='Apparent source MASK mislabel; bare mouth visible; no correction approved'
known_holds['0120.jpg:0']='Mask versus facial hair/low-quality ambiguity needs review'
manual={v['id']:v for v in json.loads((r/'audit/manual_review.json').read_text())}
for fid in ['1011.jpg:2', '0064.jpg:3', '0064.jpg:5']:
 known_holds[fid]='Prior visual audit could not verify mask state due to occlusion/reflection/quality'
# These observations are new planning metadata, not source annotation edits.
manual['1080.jpg:0']=dict(lighting='dark_background_artificial_light',pose='frontal',visible_expression='visible_smile',semantic_label_review='visually_observed')
manual['0611.jpg:0']=dict(lighting='night_outdoor',pose='three_quarter',visible_expression='not_visible',semantic_label_review='visually_observed')
manual['0045.jpg:0']=dict(lighting='dim_indoor_scene',pose='three_quarter',visible_expression='non_smiling',semantic_label_review='visually_observed')
manual['0045.jpg:4']=dict(lighting='dim_indoor_scene',pose='frontal',visible_expression='not_visible',semantic_label_review='visually_observed')
pools=defaultdict(dict);exclusions=Counter();inventory=[]
for f in faces:
 label={'face_with_mask':'MASK','face_no_mask':'NO_MASK'}.get(f['source_class'])
 reason=None
 if not f['bbox_valid']:reason='invalid_geometry'
 elif not label:reason='wearing_style_or_other_covering_requires_approved_mapping'
 elif f['id'] in known_holds:reason='known_semantic_review_hold'
 elif min(f['bbox'][2]-f['bbox'][0],f['bbox'][3]-f['bbox'][1])<64:reason='short_side_below_proposed_64px'
 row=dict(id=f['id'],source_image=f['image'],original_label=f['source_class'],planning_label=label,status=reason or 'candidate_needs_manual_review',training_eligible=False)
 inventory.append(row)
 if reason:exclusions[reason]+=1;continue
 prev=pools[label].get(f['image'])
 area=(f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1])
 if prev is None or area>prev[0]:pools[label][f['image']]=(area,f)
# Explicit metadata gets review priority; remaining candidates follow a stable hash order.
def rank(item):
 f=item[1][1]
 return (0 if f['id'] in manual and manual[f['id']]['semantic_label_review']=='visually_observed' else 1,hashlib.sha256(('reconciliation-v1:'+f['image']).encode()).hexdigest())
K=min(len(pools['MASK']),len(pools['NO_MASK']));selection=[]
for label in ['MASK','NO_MASK']:
 for _,(_,f) in sorted(pools[label].items(),key=rank)[:K]:
  m=manual.get(f['id'],{})
  selection.append(dict(face_id=f['id'],source_image=f['image'],source_sha256=next(x['sha256'] for x in images if x['path']==f['image']),original_label=f['source_class'],proposed_label=label,bbox=f['bbox'],group_id=Path(f['image']).stem,split='UNASSIGNED',lighting=m.get('lighting','unverified'),pose=m.get('pose','unverified'),visible_expression=m.get('visible_expression','unverified'),label_quality='needs_manual_verification',provenance_status='excluded_pending_original_photo_provenance',training_eligible=False,proposal_only=True))
(out/'balanced_selection_PROPOSAL.json').write_text(json.dumps(selection,indent=2))
with (out/'balanced_selection_PROPOSAL.csv').open('w',newline='') as stream:
 w=csv.DictWriter(stream,fieldnames=list(selection[0]));w.writeheader();w.writerows(selection)
(out/'face_disposition_inventory.json').write_text(json.dumps(inventory,indent=2))
summary=dict(mixed_scene_assessments=dict(Counter(x['scene_assessment'] for x in mixed)),known_semantic_holds=known_holds,short_side_minimum=64,one_candidate_per_class_per_source_image=True,candidate_unique_images_per_class={k:len(v) for k,v in pools.items()},proposed_examples_per_class=K,total_proposed_face_crops=len(selection),distinct_source_photographs=len({v['source_image'] for v in selection}),source_photographs_shared_between_classes=len({v['source_image'] for v in selection if v['proposed_label']=='MASK'} & {v['source_image'] for v in selection if v['proposed_label']=='NO_MASK'}),source_faces_by_planning_disposition=dict(exclusions),approved_training_examples=0)
(out/'planning_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
# Inspect all pages locally via an HTML index; every photograph can be opened at original size.
html=['<!doctype html><html><head><meta charset="utf-8"><title>HITL reconciliation review</title><style>body{font:16px system-ui;max-width:1800px;margin:20px auto}img{max-width:100%}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}a{color:#0645ad}</style></head><body><h1>120 mixed-annotation photographs</h1><p>Green=original MASK box; red=original NO_MASK box. These are source annotations, not corrected ground truth. All records were visually reviewed. No training set has been created.</p>']
for p in sorted(out.glob('mixed_*.jpg')):html.append(f'<h2>{p.stem}</h2><a href="{p.name}"><img loading="lazy" src="{p.name}"></a>')
html.append('<h2>Per-image observations</h2><table><tr><th>#</th><th>Image</th><th>Assessment</th><th>Observation</th></tr>')
for v in mixed:
 href='../staging_hitl_2026-11-15/extracted/'+v['image']
 html.append(f'<tr><td>{v["review_number"]}</td><td><a href="{href}">{v["filename"]}</a></td><td>{v["scene_assessment"]}</td><td>{v["review_note"]}</td></tr>')
html.append('</table><h2>Four unresolved cases</h2><img src="unresolved_four.jpg"><h2>22 images with no face-state annotations</h2><img src="no_face_state_22.jpg"></body></html>')
(out/'CONTACT_SHEET.html').write_text('\n'.join(html))
# A compact overview, while full-resolution pages remain primary evidence.
thumbs=[]
for p in sorted(out.glob('mixed_*.jpg')):
 with Image.open(p) as im:thumbs.append(ImageOps.contain(im.convert('RGB'),(450,500)))
contact=Image.new('RGB',(1800,1500),'white')
for i,im in enumerate(thumbs):contact.paste(im,((i%4)*450,(i//4)*500))
contact.save(out/'CONTACT_SHEET_ALL_120.jpg',quality=94)
