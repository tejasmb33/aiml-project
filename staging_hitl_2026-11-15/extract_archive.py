from pathlib import Path, PurePosixPath
import zipfile,json,hashlib,stat
root=Path(__file__).resolve().parent
archive=root/'raw'/'medical-mask-detection-v1-complete.zip'
record=json.loads((root/'evidence'/'archive_integrity.json').read_text())
assert record['server_md5_verified']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==record['sha256']
target=root/'extracted'
target.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    names=z.namelist()
    if len(names)!=len(set(names)): raise ValueError('Duplicate archive entry names')
    total=0
    manifest=[]
    for info in z.infolist():
        p=PurePosixPath(info.filename)
        if p.is_absolute() or '..' in p.parts or '\\' in info.filename:
            raise ValueError('Unsafe archive path')
        if stat.S_ISLNK(info.external_attr>>16): raise ValueError('Archive symlink rejected')
        total+=info.file_size
        manifest.append(dict(path=info.filename,bytes=info.file_size,crc32=f'{info.CRC:08x}'))
    if total>10*1024**3: raise ValueError('Unexpected expanded size')
    for i,info in enumerate(z.infolist()):
        dest=target/info.filename
        if info.is_dir(): dest.mkdir(parents=True,exist_ok=True);continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        with z.open(info) as src,dest.open('xb') as out:
            while chunk:=src.read(1048576): out.write(chunk)
        if (i+1)%2000==0: print(f'Extracted {i+1}/{len(names)} entries with CRC verification',flush=True)
(root/'evidence'/'archive_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Extraction complete',len(manifest),'entries,',total,'bytes',flush=True)
