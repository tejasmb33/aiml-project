from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,csv
from PIL import Image,ImageOps,ImageDraw
r=Path(__file__).resolve().parent;a=r/'audit';raw=r/'extracted'
annotations=defaultdict(list)
for p in raw.rglob('*.json'):
    obj=json.loads(p.read_text())
    if 'FileName' in obj:annotations[Path(obj['FileName']).stem].append((p,obj))
index=json.loads((r/'evidence/archive_directory_index.json').read_text())
images=[v for v in index if '/images/' in v['path']]
all_counts=Counter();unambiguous=Counter()
for stem,records in annotations.items():
    for p,obj in records:
        for ann in obj['Annotations']:
            all_counts[ann['classname']]+=1
            if len(records)==1:unambiguous[ann['classname']]+=1
inventory=dict(archive_image_entries=len(images),archive_annotation_entries=sum(map(len,annotations.values())),unique_annotation_image_stems=len(annotations),ambiguous_image_stems=sorted(k for k,v in annotations.items() if len(v)>1),annotation_records_by_class=dict(all_counts),annotation_records_by_class_excluding_ambiguous_stems=dict(unambiguous),note='Counts are source annotations, not verified usable training faces. Archive was not fully downloaded.')
(a/'full_annotation_inventory.json').write_text(json.dumps(inventory,indent=2))

# Explicit visual observations only. Unknown values are intentional.
observations = {
 '0031.jpg:11':('night_outdoor','frontal','not_visible','Night scene with artificial street lights; mouth concealed.'),
 '0121.jpg:2':('dim_indoor','downturned','not_visible','Low-lit indoor seating; face dark and blurred.'),
 '1148.jpg:0':('dim_indoor','three_quarter_image_left','not_visible','Warm artificial light in a restaurant-like interior.'),
 '0852.jpg:0':('night_outdoor','profile_image_left','not_visible','Outdoor street at night with artificial lighting.'),
 '0338.jpg:9':('daylight_shadowed','frontal','not_visible','Outdoor group; face in shadow.'),
 '1072.jpg:0':('daylight','three_quarter_image_left','not_visible','Outdoor steps, well-lit scene.'),
 '1112.jpg:0':('well_lit_indoor','profile_image_right','not_visible','Bright indoor clinical scene; glasses and hair covering.'),
 '0594.jpg:7':('well_lit_indoor','downturned','not_visible','Bright clinical scene; head angled downward.'),
 '0722.jpg:0':('daylight','frontal','not_visible','Outdoor street; spectacles and physical mask.'),
 '1179.jpg:0':('indoor_unknown_light','downturned','not_visible','Indoor transit scene; spectacles and downward pose.'),
 '1003.jpg:10':('daylight_shadowed','downturned','unknown','Outdoor scene; facial hair; no mask visible.'),
 '0634.jpg:0':('shadowed_unknown_environment','frontal','neutral_appearing','Low-key portrait with black background; dark room not established.'),
 '0391.jpg:0':('well_lit_indoor','frontal','visible_smile','Indoor portrait; visible teeth and facial hair.'),
 '0437.jpg:1':('daylight','three_quarter_image_left','open_mouth_smile','Outdoor garden; open-mouth smile. Laughing/talking cannot be distinguished from a still.'),
 '0447.jpg:0':('daylight','frontal','neutral_appearing','Outdoor portrait with facial hair.'),
 '0585.jpg:0':('daylight','frontal','visible_smile','Outdoor portrait; slight visible smile.'),
 '1171.jpg:0':('unknown','frontal_tilted','visible_smile','Pronounced in-plane tilt; smiling with teeth visible.'),
 '0178.jpg:4':('daylight','profile_image_left','neutral_appearing','Outdoor seated group; side view.'),
 '0353.jpg:7':('daylight','frontal','neutral_appearing','Outdoor group; expression readable but soft focus.'),
 '0038.jpg:13':('dim_indoor','three_quarter_image_right','unknown','Bare nose and mouth visible; source label face_with_mask appears incorrect.'),
 '0696.jpg:0':('shadowed_unknown_environment','frontal','neutral_appearing','Mouth and nose uncovered; map to NO_MASK under proposed presence policy; original improper-wearing label needs review.'),
 '1011.jpg:2':('well_lit_indoor','frontal','not_visible','Mouth hidden behind airplane seat, not a visible mask; class cannot be visually verified.'),
 '0064.jpg:3':('indoor_unknown_light','unknown','unknown','Very low-quality face/reflection-like region; unsuitable for reliable class review.'),
 '0064.jpg:5':('indoor_unknown_light','unknown','unknown','Very low-quality reflection/background region; cannot confidently verify a face or mask state.')
}
face_rows=json.loads((a/'faces.json').read_text());by_id={f['id']:f for f in face_rows}
review=[]
for id,(light,pose,expression,note) in observations.items():
    f=dict(by_id[id]);f.update(lighting=light,pose=pose,visible_expression=expression,review_note=note,review_method='visual review of context and face crop',semantic_label_review='visually_observed',training_eligible=False,eligibility_reason='Excluded: original-photo URL/author/license chain unavailable')
    if id=='0038.jpg:13':f.update(semantic_label_review='apparent_wrong_mask_label',proposed_binary_label='NO_MASK')
    elif id=='0696.jpg:0':f.update(semantic_label_review='improper_style_requires_mapping_review',proposed_binary_label='NO_MASK')
    elif id in {'1011.jpg:2','0064.jpg:3','0064.jpg:5'}:f.update(semantic_label_review='cannot_verify_due_to_occlusion_or_quality')
    review.append(f)
(a/'manual_review.json').write_text(json.dumps(review,indent=2))
for f in face_rows:
    f['training_eligible']=False
    f['eligibility_reason']='Excluded: original-photo URL/author/license chain unavailable'
    if f['id'] in observations:
        obs=next(x for x in review if x['id']==f['id'])
        for key in ('lighting','pose','visible_expression','semantic_label_review','review_note'):f[key]=obs[key]
(a/'faces.json').write_text(json.dumps(face_rows,indent=2))
summary=json.loads((a/'summary.json').read_text())
summary.update(audit_scope='partial image audit; provenance-gate exclusion stopped acquisition',manually_metadata_reviewed_faces=len(review),training_eligible_faces=0,complete_archive_downloaded=False,unreceived_image_files=len(images)-summary['image_files'])
(a/'summary.json').write_text(json.dumps(summary,indent=2))
# Show confirmed observations at readable size, with source IDs; no generated imagery.
chosen=['0031.jpg:11','1148.jpg:0','1072.jpg:0','1112.jpg:0','1003.jpg:10','0634.jpg:0','0391.jpg:0','0437.jpg:1','1171.jpg:0','0178.jpg:4','0038.jpg:13','0696.jpg:0']
sheet=Image.new('RGB',(4*420,3*360),'white');draw=ImageDraw.Draw(sheet)
for i,id in enumerate(chosen):
    f=next(x for x in review if x['id']==id)
    with Image.open(raw/f['image']) as im:
        im=im.convert('RGB');full=ImageOps.contain(im,(155,270));crop=ImageOps.contain(im.crop(tuple(f['bbox'])),(255,270))
    x,y=(i%4)*420,(i//4)*360
    sheet.paste(full,(x,y+30));sheet.paste(crop,(x+160,y+30))
    draw.text((x+3,y+3),id+' '+f['source_class'],fill='black')
    draw.text((x+3,y+310),f['lighting'],fill='black')
    draw.text((x+3,y+330),f['pose']+' / '+f['visible_expression'],fill='black')
sheet.save(a/'representative_samples.jpg',quality=93)
coverage={}
for label in ['face_with_mask','face_no_mask','face_with_mask_incorrect']:
    rec=[f for f in review if f['source_class']==label]
    coverage[label]=dict(reviewed=len(rec),lighting=dict(Counter(f['lighting'] for f in rec)),pose=dict(Counter(f['pose'] for f in rec)),visible_expression=dict(Counter(f['visible_expression'] for f in rec)))
(a/'reviewed_coverage_counts.json').write_text(json.dumps(coverage,indent=2))
(a/'issues_reviewed.json').write_text(json.dumps([f for f in review if f['semantic_label_review']!='visually_observed'],indent=2))
# Verify preservation of every existing project file captured before staging.
before=json.loads((r/'evidence/existing_files_before.json').read_text())
changed=[p for p,h in before.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
(r/'evidence/existing_files_verification.json').write_text(json.dumps(dict(existing_files_checked=len(before),changed_or_missing=changed),indent=2))
print(json.dumps(dict(archive_inventory=inventory,subset_summary=summary,existing_files_changed=changed),indent=2))
