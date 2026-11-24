"""Retrieve publisher archive with bounded concurrent range requests."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess, json, hashlib, time
root=Path(__file__).resolve().parent
size=768818472
chunk=96000000
source='https://www.kaggle.com/api/v1/datasets/download/humansintheloop/medical-mask-detection?datasetVersionNumber=1'
parts=root/'raw'/'parts'
parts.mkdir(exist_ok=True)
# Reuse only a complete prefix already obtained by the original request.
prefix=root/'raw'/'medical-mask-detection-v1.zip'
if prefix.stat().st_size>=chunk:
    with prefix.open('rb') as inp,(parts/'00.part').open('xb') as out:
        remaining=chunk
        while remaining:
            data=inp.read(min(1048576,remaining));out.write(data);remaining-=len(data)

def get(i):
    a=i*chunk;b=min(size,(i+1)*chunk)-1
    dest=parts/f'{i:02d}.part'
    if dest.exists() and dest.stat().st_size==b-a+1:
        print(f'part {i}: reused {b-a+1} bytes',flush=True);return
    cmd=['curl','-fLsS','--retry','2','--max-time','600','--range',f'{a}-{b}','-o',str(dest),source]
    subprocess.run(cmd,check=True)
    if dest.stat().st_size!=b-a+1: raise RuntimeError(f'Range size mismatch {i}')
    print(f'part {i}: verified length {b-a+1}',flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(get,range((size+chunk-1)//chunk)))
sha=hashlib.sha256();md5=hashlib.md5()
output=root/'raw'/'medical-mask-detection-v1-complete.zip'
with output.open('xb') as stream:
    for i in range((size+chunk-1)//chunk):
        with (parts/f'{i:02d}.part').open('rb') as inp:
            while data:=inp.read(1048576): stream.write(data);sha.update(data);md5.update(data)
actual=md5.hexdigest()
if actual!='5daf20b1f87cda815cc199a6c5950cc6': raise RuntimeError('Archive server MD5 mismatch')
(root/'evidence'/'archive_integrity.json').write_text(json.dumps(dict(bytes=output.stat().st_size,sha256=sha.hexdigest(),md5=actual,server_md5_verified=True,source=source),indent=2))
print('Complete archive verified against publisher storage MD5',flush=True)
