#!/usr/bin/env python3
"""Explicit bounded download of the pinned public CodeLlama7B validation checkpoint.

Never called by engine construction, CLI generation or tests. Streams two shards
with SHA256 verification; incomplete downloads can resume using validated HTTP
Range responses. No checkpoint metadata is altered.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen

MODEL='codellama/CodeLlama-7b-hf'
REVISION='6c284d1468fe6c413cf56183e69b194dcfa27fe6'
NAMES={'config.json','generation_config.json','model.safetensors.index.json',
       'model-00001-of-00002.safetensors','model-00002-of-00002.safetensors',
       'special_tokens_map.json','tokenizer.json','tokenizer.model','tokenizer_config.json'}
MAX_TOTAL_BYTES=15_000_000_000


def get_json(url):
    with urlopen(url,timeout=60) as stream:return json.load(stream)


def download_file(dest,entry):
    name=entry['rfilename'];size=entry['size'];expected=entry.get('lfs',{}).get('sha256')
    path=dest/name;partial=dest/(name+'.partial');url=f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
    begin=time.monotonic()
    if path.is_file() and path.stat().st_size==size:
        with path.open('rb') as stream:
            h=sha256()
            for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
        digest=h.hexdigest()
        if expected is None or digest==expected:return {'name':name,'bytes':size,'sha256':digest,'url':url,'reused_verified':True}
        raise ValueError(f'Existing checkpoint file checksum mismatch: {name}')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>size:raise ValueError(f'Oversized partial file: {name}')
    print('Downloading',name,'size',size,'resume',offset,flush=True)
    h=sha256()
    if offset:
        with partial.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    if offset==size and offset:
        digest=h.hexdigest()
        if expected is not None and digest!=expected:raise ValueError(f'Complete partial-file SHA256 mismatch: {name}')
        partial.replace(path)
        return {'name':name,'bytes':size,'sha256':digest,'url':url,'resumed_complete_partial':True}
    request=Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
    with urlopen(request,timeout=60) as response:
        if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
            raise ValueError('Server did not acknowledge the exact partial-file byte range')
        n=offset
        with partial.open('ab' if offset else 'wb') as stream:
            while True:
                block=response.read(1024**2)
                if not block:break
                n+=len(block)
                if n>size:raise ValueError(f'Refusing payload beyond pinned metadata: {name}')
                stream.write(block);h.update(block)
        if n!=size:raise ValueError(f'Truncated payload: {name}: {n}/{size}')
    digest=h.hexdigest()
    if expected is not None and digest!=expected:raise ValueError(f'SHA256 mismatch: {name}')
    partial.replace(path)
    elapsed=time.monotonic()-begin
    print('Completed',name,n,round(elapsed,3),digest,flush=True)
    return {'name':name,'bytes':n,'sha256':digest,'expected_lfs_sha256':expected,'url':url,'seconds':elapsed}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--receipt',required=True)
    args=parser.parse_args()
    dest=Path(args.output).resolve();dest.mkdir(parents=True,exist_ok=True)
    metadata=get_json(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true')
    if metadata.get('sha')!=REVISION or metadata.get('gated') not in (False,None):
        raise ValueError('Expected nongated pinned public checkpoint')
    files=[entry for entry in metadata['siblings'] if entry['rfilename'] in NAMES]
    if {entry['rfilename'] for entry in files}!=NAMES:raise ValueError('Validation checkpoint files missing')
    total=sum(entry['size'] for entry in files)
    if total>MAX_TOTAL_BYTES:raise ValueError('Refusing checkpoint larger than15GB')
    remaining=0
    for entry in files:
        final=dest/entry['rfilename'];partial=dest/(entry['rfilename']+'.partial')
        present=final.stat().st_size if final.is_file() else partial.stat().st_size if partial.is_file() else 0
        if present>entry['size']:raise ValueError('Existing file exceeds pinned payload size')
        remaining+=entry['size']-present
    if shutil.disk_usage(dest).free<remaining+2*1024**3:raise OSError('Insufficient free storage plus2GiBheadroom')
    (dest/'hub-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    # Read and inspect original config before any giant shard is transferred.
    config=next(entry for entry in files if entry['rfilename']=='config.json')
    results=[download_file(dest,config)]
    config_json=json.loads((dest/'config.json').read_text())
    if config_json.get('model_type')!='llama' or config_json.get('rope_scaling') or config_json.get('quantization_config'):
        raise ValueError('Checkpoint requires an unvalidated architecture/format')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results+=list(pool.map(lambda entry:download_file(dest,entry),[entry for entry in files if entry is not config]))
    receipt={'model_id':MODEL,'revision':REVISION,'total_downloaded_file_bytes':total,'max_allowed_bytes':MAX_TOTAL_BYTES,
        'config_unchanged':True,'files':results,'complete':True,'generation_validated':False,
        'license_url':f'https://huggingface.co/{MODEL}/blob/{REVISION}/LICENSE'}
    Path(args.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
    print('COMPLETE',total,flush=True)


if __name__=='__main__':main()
