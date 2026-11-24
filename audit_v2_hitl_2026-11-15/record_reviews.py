from pathlib import Path
from collections import Counter,defaultdict
import json
from PIL import Image,ImageOps,ImageDraw
out=Path(__file__).resolve().parent;r=Path('staging_hitl_2026-11-15')
records=json.loads((out/'improper_review_index.json').read_text())
no={1,2,9,12,13,14,15,16,17,18,28,30,31,34,47,48,50,52,53}
hold={3,23,27,45};invalid={5,25}
notes={
1:'Mask below mouth/chin; mouth exposed.',2:'Mask under chin; mouth exposed.',3:'Small oblique face; exact mouth coverage needs closer second review.',
4:'Nose exposed; mouth covered.',5:'Zero-area point; same photograph already has a valid masked-face annotation.',6:'Nose exposed; mouth covered.',
7:'Mouth covered; looks properly worn in this still, illustrating source improper-label ambiguity.',8:'Mouth covered while hand touches mask.',9:'Mask under chin; mouth exposed.',
10:'Nose exposed; mouth covered.',11:'Mouth covered; improper-wearing distinction is not needed for presence classification.',12:'Mask hangs from ear; mouth exposed.',
13:'Mask on forehead; mouth exposed.',14:'Mask below mouth while smoking.',15:'Mask below mouth; hand partly occludes face.',16:'Nose and mouth uncovered; mask-like straps/neck material do not establish mouth coverage.',
17:'Mask under chin; mouth exposed.',18:'Mask over eyes; mouth exposed.',19:'Masks stacked over face; mouth covered.',20:'Nose exposed; mouth covered.',21:'Nose exposed; mouth covered.',22:'Mask over mouth; soft-focus face.',
23:'Black covering may be scarf/balaclava; physical mask versus other covering unclear.',24:'Nose exposed; mouth covered.',25:'Zero-area point; existing valid annotation is face_other_covering/scarf_bandana, not a medical-mask face.',
26:'Hand adjusting mask; mouth covered.',27:'Small side view; mask appears lowered but exact lip coverage uncertain.',28:'Mask under chin; mouth exposed.',29:'Nose exposed; mouth covered.',30:'Mask under chin; mouth exposed.',
31:'Mask under chin; mouth exposed.',32:'Nose exposed; mask covers mouth.',33:'Nose exposed; mouth covered.',34:'Mask under chin; mouth visibly open.',35:'Mouth covered; face small and soft-focus.',36:'Nose exposed; mouth covered.',
37:'Mouth covered.',38:'Nose exposed; mouth covered.',39:'Printed teeth/mouth design is on a physical mask, not an exposed mouth.',40:'Mouth covered.',41:'Mouth covered; low-resolution profile.',42:'Mouth covered.',
43:'Nose exposed; mouth covered.',44:'Mouth covered; low-resolution side view.',45:'Small profile with low mask; exact mouth coverage uncertain.',46:'Mouth covered.',47:'Mask under chin; mouth exposed.',48:'Mask under chin; mouth exposed.',
49:'Mouth covered; low-resolution background face.',50:'Mask under chin; mouth exposed.',51:'Nose exposed; mouth covered.',52:'Mask under chin; mouth exposed.',53:'Mask below mouth; mouth open.',54:'Mask over mouth while being adjusted.',55:'Mask over mouth during nasal swab; nose exposed.',56:'Mouth covered.'}
for x in records:
 number=x['review_number']
 x['proposed_binary_label']='NO_MASK' if number in no else ('MASK' if number not in hold|invalid else None)
 x['audit_status']='invalid_geometry' if number in invalid else ('needs_second_review' if number in hold else 'proposed_mapping_only')
 x['visual_observation']=notes[number];x['original_annotation_modified']=False
(out/'improper_proposals.json').write_text(json.dumps(records,indent=2))
counts=dict(Counter(x['proposed_binary_label'] or x['audit_status'] for x in records))
(out/'improper_proposal_counts.json').write_text(json.dumps(counts,indent=2))

# Valid counterparts are observations, never replacements applied to the source.
invalids=json.loads((r/'audit/invalid_boxes.json').read_text())
counterparts=[1,1,2,1,6,6,6,1,1,1,1,1,1,2]
results=[]
for num,(v,counterpart) in enumerate(zip(invalids,counterparts),1):
 stem=Path(v['image']).stem
 annfiles=list((r/'extracted/Medical mask/Medical Mask/annotations').glob(stem+'.*.json'))
 assert len(annfiles)==1
 obj=json.loads(annfiles[0].read_text());other=obj['Annotations'][counterpart]
 old=obj['Annotations'][v['index']]
 result=dict(v,review_number=num,problem='zero width and height; point only',automatic_geometric_correction_safe=False,existing_valid_counterpart_index=counterpart,existing_valid_counterpart_label=other['classname'],existing_valid_counterpart_bbox=other['BoundingBox'],class_conflict=other['classname']!=v['label'],proposal='Keep existing valid annotation; flag this zero-area record as redundant/possibly accidental for approval. Do not expand into a duplicate face crop.',applied=False)
 results.append(result)
(out/'invalid_box_reviews.json').write_text(json.dumps(results,indent=2))
# Representative proposed mappings, with context and original class retained.
chosen=[1,4,13,18,19,39,55,23,5]
sheet=Image.new('RGB',(1500,1500),'white');draw=ImageDraw.Draw(sheet)
for j,num in enumerate(chosen):
 f=records[num-1];x,y=(j%3)*500,(j//3)*500
 with Image.open(r/'extracted'/f['image']) as im:
  rgb=im.convert('RGB');full=ImageOps.contain(rgb,(490,210));sheet.paste(full,(x,y+50))
  if f['bbox_valid']:
   crop=ImageOps.contain(rgb.crop(tuple(f['bbox'])),(490,210));sheet.paste(crop,(x,y+270))
 draw.text((x+5,y+3),f"#{num} {f['id']} proposed={f['proposed_binary_label'] or 'UNRESOLVED'}",fill='black')
 draw.text((x+5,y+23),f['audit_status'],fill='black')
sheet.save(out/'improper_representative_proposals.jpg',quality=94)
print('Improper mappings',counts)
print('Invalid counterpart class conflicts',[x['review_number'] for x in results if x['class_conflict']])
