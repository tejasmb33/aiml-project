from pathlib import Path
from collections import Counter,defaultdict
from PIL import Image,ImageOps,ImageDraw
import json,hashlib,csv,math
out=Path(__file__).resolve().parent;r=Path('staging_hitl_2026-11-15');raw=r/'extracted'
prior={}
for p in Path('.').rglob('*'):
 if any(x in {'.venv','.git','__pycache__','.pytest_cache',out.name} for x in p.parts):continue
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   while b:=f.read(1048576):h.update(b)
  prior[str(p)]=h.hexdigest()
(out/'preservation_before.json').write_text(json.dumps(prior,indent=2))
faces=json.loads((r/'audit/faces.json').read_text());images=json.loads((r/'audit/images.json').read_text())
by_image=defaultdict(list)
for f in faces:by_image[f['image']].append(f)

def write(name,obj):(out/name).write_text(json.dumps(obj,indent=2))
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
 intersection=max(0,x2-x1)*max(0,y2-y1)
 union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
 return intersection/union if union else 0
ledger=[];mixed=[];neither=[];overlaps=[]
for im in images:
 rows=by_image[im['path']];counts=Counter(f['source_class'] for f in rows)
 m=[f for f in rows if f['source_class']=='face_with_mask'];n=[f for f in rows if f['source_class']=='face_no_mask']
 category='both' if m and n else ('MASK_without_NO_MASK' if m else ('NO_MASK_without_MASK' if n else 'neither'))
 record=dict(image=im['path'],filename=im['filename'],category=category,face_counts=dict(counts),valid_face_records=sum(f['bbox_valid'] for f in rows))
 ledger.append(record)
 if category=='both':
  pairs=[]
  for a in m:
   for b in n:
    if a['bbox_valid'] and b['bbox_valid']:
     overlap=iou(a['bbox'],b['bbox'])
     if overlap>=.3:pairs.append(dict(mask_id=a['id'],no_mask_id=b['id'],iou=overlap))
  rec=dict(record,review_number=len(mixed)+1,mask_ids=[f['id'] for f in m],no_mask_ids=[f['id'] for f in n],cross_class_overlap_candidates=pairs,valid_mask_count=sum(f['bbox_valid'] for f in m),valid_no_mask_count=sum(f['bbox_valid'] for f in n),visual_review='pending')
  mixed.append(rec);overlaps+=pairs
 elif category=='neither':
  o=counts.get('face_other_covering',0)>0;i=counts.get('face_with_mask_incorrect',0)>0
  record['neither_subgroup']='OTHER_COVERING_and_IMPROPER' if o and i else ('OTHER_COVERING_only' if o else ('IMPROPER_only' if i else 'no_face_state_annotation'))
  neither.append(record)
write('image_ledger.json',ledger);write('mixed_images.json',mixed);write('neither_images.json',neither);write('overlap_candidates.json',overlaps)
write('reconciliation_counts.json',dict(total_images=len(images),partitions=dict(Counter(x['category'] for x in ledger)),neither_subgroups=dict(Counter(x['neither_subgroup'] for x in neither)),mixed_cross_class_overlap_pairs=len(overlaps)))
with (out/'image_ledger.csv').open('w',newline='') as stream:
 w=csv.DictWriter(stream,fieldnames=['image','filename','category','valid_face_records','face_counts']);w.writeheader();w.writerows([{k:v for k,v in row.items() if k in w.fieldnames} for row in ledger])
# Every mixed photograph: original context with all valid class boxes, plus representative crops.
for start in range(0,len(mixed),10):
 sheet=Image.new('RGB',(1800,2000),'white');draw=ImageDraw.Draw(sheet)
 for j,record in enumerate(mixed[start:start+10]):
  x,y=(j%2)*900,(j//2)*400;rows=by_image[record['image']]
  with Image.open(raw/record['image']) as im:
   rgb=im.convert('RGB');thumb=ImageOps.contain(rgb,(575,345));sx=thumb.width/rgb.width;sy=thumb.height/rgb.height;td=ImageDraw.Draw(thumb)
   for f in rows:
    if f['source_class'] not in {'face_with_mask','face_no_mask'}:continue
    color='green' if f['source_class']=='face_with_mask' else 'red'
    if f['bbox_valid']:
     box=[f['bbox'][0]*sx,f['bbox'][1]*sy,f['bbox'][2]*sx,f['bbox'][3]*sy];td.rectangle(box,outline=color,width=2)
     td.text((box[0],max(0,box[1]-10)),str(f['index']),fill=color)
   sheet.paste(thumb,(x,y+45))
   for c,label in enumerate(['face_with_mask','face_no_mask']):
    valid=[f for f in rows if f['source_class']==label and f['bbox_valid']]
    if valid:
     # Prefer a cross-label overlap candidate, otherwise largest face.
     matches=[p['mask_id' if c==0 else 'no_mask_id'] for p in record['cross_class_overlap_candidates']]
     face=next((f for f in valid if f['id'] in matches),max(valid,key=lambda f:(f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1])))
     crop=ImageOps.contain(rgb.crop(tuple(face['bbox'])),(310,155));sheet.paste(crop,(x+585,y+45+c*175));draw.text((x+585,y+25+c*175),face['id']+' '+('MASK' if c==0 else 'NO_MASK'),fill='black')
    else:draw.text((x+585,y+50+c*175),'NO VALID BOX for '+label,fill='red')
  draw.text((x+4,y+3),f"#{record['review_number']} {record['filename']} M={len(record['mask_ids'])} N={len(record['no_mask_ids'])} overlaps={len(record['cross_class_overlap_candidates'])}",fill='black')
 sheet.save(out/f'mixed_{start//10+1:02d}.jpg',quality=94)
# All 187 neither-main-label images, for traceability (no forced binary label).
for start in range(0,len(neither),24):
 sheet=Image.new('RGB',(1800,1500),'white');d=ImageDraw.Draw(sheet)
 for j,v in enumerate(neither[start:start+24]):
  x,y=(j%6)*300,(j//6)*375
  with Image.open(raw/v['image']) as im:thumb=ImageOps.contain(im.convert('RGB'),(290,310))
  sheet.paste(thumb,(x,y+55));d.text((x+3,y+3),v['filename'],fill='black');d.text((x+3,y+20),v['neither_subgroup'],fill='black')
 sheet.save(out/f'neither_{start//24+1:02d}.jpg',quality=90)
# Four uncertain improper-wearing records, full context and an enlarged crop.
proposals=json.loads(Path('audit_v2_hitl_2026-11-15/improper_proposals.json').read_text())
unresolved=[f for f in proposals if f['audit_status']=='needs_second_review']
sheet=Image.new('RGB',(1600,1500),'white');d=ImageDraw.Draw(sheet)
for j,f in enumerate(unresolved):
 x,y=(j%2)*800,(j//2)*750
 with Image.open(raw/f['image']) as im:
  rgb=im.convert('RGB');full=ImageOps.contain(rgb,(780,350));crop=ImageOps.contain(rgb.crop(tuple(f['bbox'])),(780,350))
 sheet.paste(full,(x,y+40));sheet.paste(crop,(x,y+395));d.text((x+5,y+3),f['id']+' '+str(f['bbox']),fill='black')
sheet.save(out/'unresolved_four.jpg',quality=96)
write('unresolved_four_source.json',unresolved)
print((out/'reconciliation_counts.json').read_text());print('Overlap candidates',json.dumps(overlaps))
