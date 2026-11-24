from pathlib import Path
from collections import Counter,defaultdict
import struct,json,hashlib
from PIL import Image,ImageOps,ImageDraw
r=Path('staging_hitl_2026-11-15');out=Path(__file__).resolve().parent
raw=r/'extracted'; old=r/'audit'
images=json.loads((old/'images.json').read_text()); faces=json.loads((old/'faces.json').read_text())
# Parse the full ZIP central directory, including true ZIP64 lengths.
b=(r/'evidence/complete_directory_tail.bin').read_bytes();base=766818472
at=b.rfind(b'PK\x05\x06');e=struct.unpack('<4s4H2IH',b[at:at+22]);pos=e[6]-base
entries=[]
while b[pos:pos+4]==b'PK\x01\x02':
 h=struct.unpack('<4s6H3I5H2I',b[pos:pos+46]);n,x,c=h[10:13]
 name=b[pos+46:pos+46+n].decode();extra=b[pos+46+n:pos+46+n+x]
 cs,us,offset=h[8],h[9],h[-1];k=0
 while k<len(extra):
  eid,length=struct.unpack('<HH',extra[k:k+4]);data=extra[k+4:k+4+length];k+=4+length
  if eid==1:
   values=iter(struct.unpack('<'+'Q'*(len(data)//8),data))
   if us==0xffffffff:us=next(values)
   if cs==0xffffffff:cs=next(values)
   if offset==0xffffffff:offset=next(values)
 entries.append(dict(path=name,offset=offset,compressed_bytes=cs,uncompressed_bytes=us,crc32=f'{h[7]:08x}'))
 pos+=46+n+x+c
entries.sort(key=lambda v:v['offset'])
for i,v in enumerate(entries):v['end_offset']=entries[i+1]['offset'] if i+1<len(entries) else e[6]
intervals=[(0,(r/'raw/medical-mask-detection-v1.zip').stat().st_size)]
for p in (r/'raw/parts').glob('*.part'):
 start=int(p.stem)*96000000;intervals.append((start,start+p.stat().st_size))
intervals += [(760000000,760000000+(r/'evidence/archive_tail.bin').stat().st_size),(766818472,768818472)]
merged=[]
for start,end in sorted(intervals):
 if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
 else:merged.append([start,end])
received={x['path'] for x in images};remaining=[]
for v in entries:
 if '/images/' not in v['path'] or v['path'] in received:continue
 overlap=sum(max(0,min(v['end_offset'],hi)-max(v['offset'],lo)) for lo,hi in merged)
 full=v['end_offset']-v['offset']
 status='complete_entry_bytes_present_not_extracted' if overlap==full else ('partial_entry_bytes_present' if overlap else 'entry_bytes_absent')
 remaining.append(dict(v,status=status,available_entry_bytes=overlap,required_entry_bytes=full))
(out/'remaining_image_status.json').write_text(json.dumps(remaining,indent=2))
(out/'download_forensics.json').write_text(json.dumps(dict(expected_archive_bytes=768818472,primary_received_bytes=(r/'raw/medical-mask-detection-v1.zip').stat().st_size,primary_stop='incomplete entry 1271.jpg',decoded_images=len(images),not_decoded=len(remaining),remaining_categories=dict(Counter(v['status'] for v in remaining)),available_intervals=merged),indent=2))

# Separate overlapping presence labels from exclusive photograph categories.
class_sets=defaultdict(set);valid_class_sets=defaultdict(set);by_image=defaultdict(list)
for f in faces:
 class_sets[f['source_class']].add(f['image']);by_image[f['image']].append(f)
 if f['bbox_valid']:valid_class_sets[f['source_class']].add(f['image'])
m=class_sets['face_with_mask'];n=class_sets['face_no_mask'];all_images={x['path'] for x in images}
results=dict(total_unique_images=len(all_images),MASK_images=len(m),NO_MASK_images=len(n),both_MASK_and_NO_MASK=len(m&n),MASK_but_no_NO_MASK=len(m-n),NO_MASK_but_no_MASK=len(n-m),neither_main_source_label=len(all_images-(m|n)),at_least_one_main_label=len(m|n),source_class_image_counts={k:len(v) for k,v in class_sets.items()},valid_box_class_image_counts={k:len(v) for k,v in valid_class_sets.items()},note='Source annotation presence per photograph; not manually verified binary whole-image labels.')
(out/'unique_image_balance.json').write_text(json.dumps(results,indent=2))

# Review all 56 improper annotations with original context and nearest useful crop.
improper=[f for f in faces if f['source_class']=='face_with_mask_incorrect']
for start in range(0,len(improper),6):
 selected=improper[start:start+6];sheet=Image.new('RGB',(1500,1080),'white');d=ImageDraw.Draw(sheet)
 for j,f in enumerate(selected):
  x,y=(j%3)*500,(j//3)*540
  with Image.open(raw/f['image']) as im:
   rgb=im.convert('RGB');full=ImageOps.contain(rgb,(490,240));sheet.paste(full,(x,y+30))
   if f['bbox_valid']:
    crop=ImageOps.contain(rgb.crop(tuple(f['bbox'])),(490,240));sheet.paste(crop,(x,y+280))
   else:
    d.text((x+10,y+320),'ZERO-AREA BOX: no crop exists',fill='red')
  d.text((x+4,y+4),f"#{start+j+1} {f['id']} {f['bbox']}",fill='black')
 sheet.save(out/f'improper_{start//6+1:02d}.jpg',quality=94)
(out/'improper_review_index.json').write_text(json.dumps([dict(f,review_number=i+1,sheet=f'improper_{i//6+1:02d}.jpg',cell=i%6+1) for i,f in enumerate(improper)],indent=2))

invalid=json.loads((old/'invalid_boxes.json').read_text())
for start in range(0,len(invalid),4):
 sheet=Image.new('RGB',(1600,1100),'white');d=ImageDraw.Draw(sheet)
 for j,v in enumerate(invalid[start:start+4]):
  x,y=(j%2)*800,(j//2)*550
  with Image.open(raw/v['image']) as im:
   rgb=im.convert('RGB'); thumb=ImageOps.contain(rgb,(780,480));sx,sy=thumb.width/rgb.width,thumb.height/rgb.height
   td=ImageDraw.Draw(thumb);px,py=v['bbox'][0]*sx,v['bbox'][1]*sy
   td.line((px-12,py,px+12,py),fill='red',width=3);td.line((px,py-12,px,py+12),fill='red',width=3)
   sheet.paste(thumb,(x+5,y+50))
  d.text((x+5,y+5),f"#{start+j+1} {Path(v['image']).name}:{v['index']} {v['label']}",fill='black')
  d.text((x+5,y+25),f"box={v['bbox']}; red cross=only recorded point",fill='black')
 sheet.save(out/f'invalid_boxes_{start//4+1:02d}.jpg',quality=94)
print(json.dumps(dict(download=json.loads((out/'download_forensics.json').read_text()),balance=results),indent=2))
