from pathlib import Path
from collections import Counter
import json,struct,zlib,hashlib
r=Path('staging_hitl_2026-11-15');out=Path(__file__).resolve().parent
segments=[(0,r/'raw/medical-mask-detection-v1.zip')]
segments += [(int(p.stem)*96000000,p) for p in (r/'raw/parts').glob('*.part')]
segments += [(760000000,r/'evidence/archive_tail.bin'),(766818472,r/'evidence/complete_directory_tail.bin')]

def read_at(start,length):
    result=bytearray();pos=start
    while len(result)<length:
        options=[(offset,p) for offset,p in segments if offset<=pos<offset+p.stat().st_size]
        if not options:raise ValueError('missing byte interval')
        offset,p=max(options,key=lambda v:v[0]+v[1].stat().st_size)
        with p.open('rb') as f:
            f.seek(pos-offset);b=f.read(min(length-len(result),offset+p.stat().st_size-pos))
        if not b:raise ValueError('unexpected EOF')
        result.extend(b);pos+=len(b)
    return bytes(result)
remaining=json.loads((out/'remaining_image_status.json').read_text());checked=[]
for v in remaining:
    if v['status']!='complete_entry_bytes_present_not_extracted':continue
    result=dict(path=v['path'],crc32_verified=False)
    try:
        h=struct.unpack('<4s5H3I2H',read_at(v['offset'],30))
        assert h[0]==b'PK\x03\x04'
        data=read_at(v['offset']+30+h[-2]+h[-1],v['compressed_bytes'])
        body=zlib.decompress(data,-15) if h[3]==8 else data
        assert len(body)==v['uncompressed_bytes'] and f'{zlib.crc32(body):08x}'==v['crc32']
        result['crc32_verified']=True
    except Exception as exc:result['error']=str(exc)
    checked.append(result)
(out/'complete_fragment_entries_crc.json').write_text(json.dumps(checked,indent=2))
faces=json.loads((r/'audit/faces.json').read_text());screen={}
for label in ['face_with_mask','face_no_mask']:
    valid=[f for f in faces if f['source_class']==label and f['bbox_valid']]
    dark=[f for f in valid if f['luma_mean']<=60]
    bright=[f for f in valid if f['luma_mean']>=130]
    dark_imgs={f['image'] for f in dark};bright_imgs={f['image'] for f in bright}
    screen[label]=dict(valid_faces=len(valid),unique_images=len({f['image'] for f in valid}),dark_candidate_faces=len(dark),dark_candidate_unique_images=len(dark_imgs),bright_candidate_faces=len(bright),bright_candidate_unique_images=len(bright_imgs),both_candidate_types_same_image=len(dark_imgs&bright_imgs),threshold_note='8-bit grayscale crop mean <=60 or >=130; screening only, NOT actual illumination labels',dark_ids=[f['id'] for f in dark],bright_ids=[f['id'] for f in bright])
(out/'lighting_proxy_screen.json').write_text(json.dumps(screen,indent=2))
print('complete retained fragment entries CRC verified',sum(x['crc32_verified'] for x in checked),'of',len(checked))
print(json.dumps({k:{a:b for a,b in v.items() if not a.endswith('_ids')} for k,v in screen.items()},indent=2))
