import hashlib,json,os,sys,time
from pathlib import Path
model=Path(sys.argv[1]);output=Path(sys.argv[2]);before=model.stat();digest=hashlib.sha256();total=0;begin=time.monotonic()
with model.open('rb',buffering=0) as stream:
    while chunk:=stream.read(4<<20):digest.update(chunk);total+=len(chunk)
after=model.stat()
result={'model_path':str(model),'bytes_hashed':total,'sha256':digest.hexdigest(),'elapsed_seconds':time.monotonic()-begin,
    'read_only':True,'chunk_bytes':4<<20,'before':{'inode':before.st_ino,'bytes':before.st_size,'mtime_ns':before.st_mtime_ns},
    'after':{'inode':after.st_ino,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}}
assert result['before']==result['after'] and total==before.st_size
assert digest.hexdigest()=='f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d'
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
