"""Recover only fully received, CRC-verified local ZIP entries; not a complete archive."""
from pathlib import Path,PurePosixPath
import struct,zlib,json,hashlib
r=Path(__file__).resolve().parent
source=r/'raw'/'medical-mask-detection-v1.zip'
target=r/'extracted';target.mkdir(exist_ok=False)
manifest=[];stop=None
with source.open('rb') as f:
    while True:
        offset=f.tell();h=f.read(30)
        if len(h)<30 or h[:4]!=b'PK\x03\x04':stop='end of received prefix';break
        v=struct.unpack('<4s5H3I2H',h)
        name=f.read(v[-2]).decode();extra=f.read(v[-1]);cs,us=v[7:9]
        if cs==0xffffffff or us==0xffffffff:
            k=0
            while k<len(extra):
                eid,n=struct.unpack('<HH',extra[k:k+4]);data=extra[k+4:k+4+n];k+=4+n
                if eid==1:
                    vals=iter(struct.unpack('<'+'Q'*(len(data)//8),data))
                    if us==0xffffffff:us=next(vals)
                    if cs==0xffffffff:cs=next(vals)
                    break
        payload=f.read(cs)
        if len(payload)!=cs:stop=f'incomplete entry: {name}';break
        body=zlib.decompress(payload,-15) if v[3]==8 else payload
        if len(body)!=us or zlib.crc32(body)!=v[6]:raise ValueError(f'CRC/size mismatch: {name}')
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('unsafe path')
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(body)
        manifest.append(dict(path=name,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),crc32_verified=True))
# The range-verified tail contains the class metadata; preserve it alongside.
meta=target/'Medical mask'/'meta.json'
meta.write_bytes((r/'evidence/meta.json').read_bytes())
(r/'evidence/received_extraction.json').write_text(json.dumps(dict(complete_archive=False,entries=manifest,stop=stop),indent=2))
print('Recovered complete entries:',len(manifest),'stop:',stop)
