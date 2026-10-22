#!/usr/bin/env python3
"""Rebuild patch, integrity manifest and distributable without scratch/build data."""
import argparse,difflib,hashlib,json
from pathlib import Path
import zipfile

p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path);p.add_argument('--output',type=Path,default=Path('AirLLM-Next-2026-10-22.zip'));a=p.parse_args()
root=Path(__file__).resolve().parents[1]

def include(path):
    r=path.relative_to(root)
    if any(x.startswith('.') or x in ('__pycache__','build') or x.endswith('.egg-info') for x in r.parts):return False
    if r.suffix=='.pyc':return False
    if any(x in r.parts for x in ('stress-metadata','wheel-meta')):return False
    if r.parts[:2]==('air_llm','dist'):return False
    return True

if a.baseline:
    base=a.baseline.resolve();diff=[];changed=[]
    paths={x.relative_to(root) for x in root.rglob('*') if x.is_file() and include(x)}|{x.relative_to(base) for x in base.rglob('*') if x.is_file()}
    for rel in sorted(paths):
        if rel.suffix not in ('.py','.sh','.txt','.md','.toml') or rel.name in ('changes-next.patch',):continue
        if any(x in ('build','dist','dist-next','__pycache__') or x.startswith('.') or x.endswith('.egg-info') for x in rel.parts):continue
        src=base/rel;dst=root/rel
        if any(x.exists() and x.stat().st_size>600000 for x in (src,dst)):continue
        old=src.read_text(errors='replace').splitlines(keepends=True) if src.exists() else []
        new=dst.read_text(errors='replace').splitlines(keepends=True) if dst.exists() else []
        if old!=new:
            diff+=difflib.unified_diff(old,new,fromfile='a/'+str(rel) if src.exists() else '/dev/null',tofile='b/'+str(rel))
            changed.append(str(rel))
    (root/'changes-next.patch').write_text(''.join(diff))
    (root/'evidence-next/changed-files.json').write_text(json.dumps(changed,indent=2))
files=[p for p in sorted(root.rglob('*')) if p.is_file() and include(p) and p.name!='archive-manifest.json']
manifest={'format_version':1,'project':'AirLLM Next','version':'4.0.0+next.1','files':[
 {'path':str(x.relative_to(root)),'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in files]}
(root/'archive-manifest.json').write_text(json.dumps(manifest,indent=2));files.append(root/'archive-manifest.json')
a.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in files:z.write(f,root.name+'/'+str(f.relative_to(root)))
with zipfile.ZipFile(a.output) as z:
    if z.testzip() is not None:raise RuntimeError('Archive CRC validation failed')
print(json.dumps({'archive':str(a.output.resolve()),'bytes':a.output.stat().st_size,'file_count':len(files),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()},indent=2))
