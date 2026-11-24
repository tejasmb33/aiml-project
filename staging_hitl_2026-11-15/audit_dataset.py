"""Read-only audit of staged originals. Writes results only below audit/."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, csv, math
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'extracted'
OUT = ROOT / 'audit'
OUT.mkdir(exist_ok=True)

def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False))

def dhash(image):
    a = np.asarray(image.convert('L').resize((9, 8)))
    return (a[:, 1:] > a[:, :-1]).ravel().astype(np.uint8)

files = sorted(p for p in RAW.rglob('*') if p.is_file())
image_ext = {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff','.gif'}
images = [p for p in files if p.suffix.lower() in image_ext]
annotations = {}
annotation_errors = []
metadata_keys = Counter()
for p in files:
    if p.suffix.lower() == '.json':
        try:
            obj = json.loads(p.read_text())
            if isinstance(obj, dict):
                metadata_keys.update(obj.keys())
                if 'Annotations' in obj and 'FileName' in obj:
                    annotations[Path(obj['FileName']).stem] = (p, obj)
        except Exception as exc:
            annotation_errors.append({'path':str(p.relative_to(RAW)), 'error':str(exc)})

rows, faces, corrupt, boxes_invalid, missing_annotations = [], [], [], [], []
classes = Counter()
formats = Counter()
byte_groups, pixel_groups = defaultdict(list), defaultdict(list)
hashes = []
for i,p in enumerate(images):
    rel = str(p.relative_to(RAW))
    try:
        binary = p.read_bytes()
        sha = hashlib.sha256(binary).hexdigest()
        with Image.open(p) as check:
            fmt = check.format
            check.verify()
        with Image.open(p) as orig:
            orig.load()
            rgb = orig.convert('RGB')
            w,h = rgb.size
            orientation = orig.getexif().get(274,1)
            pixel_sha = hashlib.sha256(f'{w}x{h}:'.encode()+rgb.tobytes()).hexdigest()
            lum = np.asarray(rgb.convert('L').resize((128,128)),dtype=np.float32)
            row = dict(path=rel,filename=p.name,format=fmt,width=w,height=h,
                       exif_orientation=orientation,sha256=sha,pixel_sha256=pixel_sha,
                       luma_mean=float(lum.mean()),luma_p10=float(np.percentile(lum,10)),
                       luma_p90=float(np.percentile(lum,90)))
            obj = annotations.get(p.stem)
            if obj is None:
                missing_annotations.append(rel)
            else:
                row['annotation_path'] = str(obj[0].relative_to(RAW))
                row['face_annotations'] = 0
                for n,ann in enumerate(obj[1].get('Annotations',[])):
                    label = ann.get('classname','UNKNOWN')
                    classes[label]+=1
                    box = ann.get('BoundingBox')
                    valid = (isinstance(box,list) and len(box)==4
                             and all(isinstance(x,(int,float)) and math.isfinite(x) for x in box)
                             and 0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h)
                    if not valid:
                        boxes_invalid.append(dict(image=rel,index=n,label=label,bbox=box,width=w,height=h))
                    if label in {'face_with_mask', 'face_no_mask', 'face_with_mask_incorrect', 'face_other_covering'}:
                        row['face_annotations']+=1
                        rec = dict(id=f'{p.name}:{n}',image=rel,index=n,source_class=label,bbox=box,
                                   bbox_valid=valid,lighting='unreviewed',pose='unreviewed',
                                   visible_expression='unreviewed',semantic_label_review='unreviewed',
                                   training_eligible=False,eligibility_reason='provenance and audit pending')
                        if valid:
                            crop = rgb.crop(tuple(map(int,box)))
                            gray = np.asarray(crop.convert('L').resize((64,64)),dtype=np.float32)
                            rec.update(luma_mean=float(gray.mean()),luma_p10=float(np.percentile(gray,10)),
                                       luma_p90=float(np.percentile(gray,90)))
                        faces.append(rec)
            rows.append(row)
            formats[fmt]+=1
            byte_groups[sha].append(rel)
            pixel_groups[pixel_sha].append(rel)
            hashes.append(dhash(rgb))
    except Exception as exc:
        corrupt.append(dict(path=rel,error=f'{type(exc).__name__}: {exc}'))
    if (i+1)%500==0:
        print(f'Fully decoded {i+1}/{len(images)} images',flush=True)

write_json('images.json',rows)
write_json('faces.json',faces)
write_json('corrupt_images.json',corrupt)
write_json('invalid_boxes.json',boxes_invalid)
write_json('annotation_errors.json',annotation_errors)
write_json('missing_annotations.json',missing_annotations)
byte_dups=[v for v in byte_groups.values() if len(v)>1]
pixel_dups=[v for v in pixel_groups.values() if len(v)>1]
write_json('exact_duplicate_groups.json',byte_dups)
write_json('pixel_duplicate_groups.json',pixel_dups)
print('Searching perceptual duplicate candidates...',flush=True)
pairs=[]
if hashes:
    arr=np.asarray(hashes,dtype=float)
    candidates=cKDTree(arr).query_pairs(r=4,p=1,output_type='ndarray')
    for a,b in candidates:
        if rows[a]['pixel_sha256'] != rows[b]['pixel_sha256']:
            pairs.append(dict(image_a=rows[a]['path'],image_b=rows[b]['path'],
                              dhash_distance=int(np.abs(arr[a]-arr[b]).sum()),status='requires_visual_review'))
write_json('near_duplicate_candidates.json',pairs)
face_counts=Counter(f['source_class'] for f in faces)
images_by_class={c:len({f['image'] for f in faces if f['source_class']==c}) for c in face_counts}
summary=dict(image_files=len(images),decoded_images=len(rows),corrupt_count=len(corrupt),
             formats=dict(formats),face_annotation_counts=dict(face_counts),
             photographs_containing_class=images_by_class,all_annotation_counts=dict(classes),
             invalid_box_count=len(boxes_invalid),missing_annotation_count=len(missing_annotations),
             annotation_error_count=len(annotation_errors),
             byte_duplicate_groups=len(byte_dups),byte_duplicate_redundant_files=sum(len(v)-1 for v in byte_dups),
             pixel_duplicate_groups=len(pixel_dups),pixel_duplicate_redundant_files=sum(len(v)-1 for v in pixel_dups),
             near_duplicate_candidate_pairs=len(pairs),json_top_level_keys=dict(metadata_keys),
             training_eligible_faces=0)
write_json('summary.json',summary)

# Deterministic, independently labelled review samples, not a population estimate.
review=[]
for label in sorted(face_counts):
    pool=sorted((f for f in faces if f['source_class']==label and f['bbox_valid']),key=lambda f:(f['luma_mean'],f['id']))
    groups={'darkest':pool[:20],'brightest':pool[-20:]}
    groups['spread']=[pool[int(i)] for i in np.linspace(0,len(pool)-1,min(24,len(pool)))] if pool else []
    for group,chosen in groups.items():
        for start in range(0,len(chosen),12):
            page=chosen[start:start+12]
            sheet=Image.new('RGB',(4*300,3*260),'#eeeeee')
            draw=ImageDraw.Draw(sheet)
            for j,f in enumerate(page):
                p=RAW/f['image']
                with Image.open(p) as im:
                    rgb=im.convert('RGB')
                    # Top-left thumbnail gives environmental context; face stays at right.
                    full=ImageOps.contain(rgb,(115,190))
                    crop=ImageOps.contain(rgb.crop(tuple(map(int,f['bbox']))),(175,190))
                x,y=(j%4)*300,(j//4)*260
                sheet.paste(full,(x,y+25));sheet.paste(crop,(x+120,y+25))
                draw.text((x+3,y+3),f['id'],fill='black')
                draw.text((x+3,y+220),f"Y={f['luma_mean']:.1f} {label}",fill='black')
                rec=dict(f)
                rec['sample_group']=group
                rec['sheet']=f'{label}_{group}_{start//12+1}.jpg'
                rec['cell']=j+1
                review.append(rec)
            sheet.save(OUT/f'{label}_{group}_{start//12+1}.jpg',quality=88)
write_json('review_sample_index.json',review)
print(json.dumps(summary,indent=2),flush=True)
