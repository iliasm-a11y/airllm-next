"""Plan, then optionally hardlink byte-identical CLOSED synthetic native captures.

Only explicitly selected native-* runs below this work directory are eligible.
No weights, user files, source files, metadata or active-process captures can be
replaced. Every original name, size and SHA256 remains verified. Shared inodes
must remain immutable after deduplication. Reported byte savings are logical
redundancy estimates, not measured APFS/device space reclamation.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid
import psutil

WORK_ROOT=Path(__file__).resolve().parent
BENCHMARK_LOCK=Path('/tmp/airllm-glm53-benchmark.lock')
MiB=1024**2


def utc():return datetime.now(timezone.utc).isoformat()
def identity(path):
    s=path.stat();return {'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'mode':stat.S_IMODE(s.st_mode),'uid':s.st_uid,'gid':s.st_gid,'links':s.st_nlink}
def data_identity(path):
    x=identity(path);return (x['device'],x['inode'],x['size'],x['mtime_ns'])
def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.chmod(0o600);temp.replace(path)
def digest(path):
    before=data_identity(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(MiB),b''):h.update(block)
    if data_identity(path)!=before:raise ValueError('capture changed while hashing: '+str(path))
    return h.hexdigest()

def check_closed(run_root,*,process_factory=psutil.Process):
    original=Path(run_root)
    run=original.resolve(strict=True)
    if original.is_symlink() or not run.is_relative_to(WORK_ROOT.resolve()) or not re.fullmatch(r'native-[A-Za-z0-9_.-]+',run.name):
        raise ValueError('only explicit native-* research runs below this work directory are allowed')
    guard_path=run/'guard.json';raw=guard_path.read_bytes();guard=json.loads(raw)
    if guard.get('status') not in ('completed','failed','cancelled','rejected') or type(guard.get('returncode')) is not int:
        raise ValueError('capture has no closed owned native-process receipt')
    owner=guard.get('owned_identity')
    if not isinstance(owner,dict) or type(owner.get('pid')) is not int or type(owner.get('create_time')) not in (int,float) or not isinstance(owner.get('executable'),str):
        raise ValueError('closed receipt needs the owned PID birth and executable')
    if owner['pid']!=guard.get('owned_pid'):
        raise ValueError('closed owned PID metadata differs')
    try:
        process=process_factory(owner['pid'])
        if process.create_time()==owner['create_time'] and process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
            raise ValueError('recorded native helper is still active; no deduplication')
    except psutil.NoSuchProcess:pass
    capture=run/'capture'
    if capture.is_symlink() or not capture.is_dir() or capture.resolve().parent!=run:
        raise ValueError('ordinary direct capture directory required')
    return run,capture,{'guard_sha256':hashlib.sha256(raw).hexdigest(),'guard_identity':identity(guard_path),'owned_identity':owner,
        'returncode':guard['returncode'],'status':guard['status'],'closed_checked_at_utc':utc()}

def json_lines(path):
    if not path.exists():return []
    if path.stat().st_size>64*MiB:raise ValueError('capture manifest exceeds64MiB')
    raw=path.read_bytes();complete=raw[:raw.rfind(b'\n')+1]
    return [json.loads(line) for line in complete.splitlines() if line.strip()]

def collect(capture):
    """Only payloads named by native capture manifests are candidates."""
    entries={}
    def add(name,size,sha256,cap):
        if not isinstance(name,str) or Path(name).name!=name or not isinstance(sha256,str) or re.fullmatch('[0-9a-f]{64}',sha256) is None:
            raise ValueError('invalid native payload identity')
        path=capture/name
        if path.is_symlink() or not path.is_file() or path.resolve().parent!=capture.resolve():
            raise ValueError('payload is missing, symlinked or outside capture: '+name)
        record=identity(path)
        if record['uid']!=os.getuid() or not 0<record['size']<=cap or (size is not None and record['size']!=size):
            raise ValueError('native payload ownership/size refused: '+name)
        previous=entries.get(str(path))
        entry={'path':str(path),'size':record['size'],'sha256':sha256,'before':record}
        if previous is not None and (previous['size'],previous['sha256'])!=(entry['size'],entry['sha256']):
            raise ValueError('conflicting native payload records: '+name)
        entries[str(path)]=entry
    for manifest in ('tensors.jsonl','continuation_tensors.jsonl'):
        for row in json_lines(capture/manifest):
            if re.fullmatch(r'tensor-\d+\.bin',str(row.get('file',''))) is None or not (str(row.get('name','')).startswith(('kda_scan_out-','ffn_moe_topk-','ffn_moe_logits-')) or row.get('name')=='result_output'):
                raise ValueError('unexpected synthetic tensor payload name')
            add(row['file'],row['bytes'],row['sha256'],64*MiB)
    report_path=capture/'report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    for row in report.get('requests',[]):
        request=row.get('request')
        if type(request) is not int or request not in (0,1,2):raise ValueError('invalid native request index')
        if 'logits_sha256' in row:add(f'logits-{request}.f32',None,row['logits_sha256'],4*MiB)
        if 'sequence_sha256' in row:add(f'final-state-{request}.bin',row['sequence_bytes'],row['sequence_sha256'],256*MiB)
        if 'continuation_state_sha256' in row:add(f'continuation-state-{request}.bin',row['continuation_state_bytes'],row['continuation_state_sha256'],256*MiB)
        for step in row.get('continuation',[]):
            index=step.get('step')
            if type(index) is not int or not 0<=index<4:raise ValueError('invalid bounded continuation step')
            add(f'continuation-logits-{request}-{index}.f32',None,step['logits_sha256'],4*MiB)
    return list(entries.values())

def same_bytes(a,b):
    before_a,before_b=data_identity(a),data_identity(b)
    if before_a[2]!=before_b[2]:return False
    equal=True
    with a.open('rb') as left,b.open('rb') as right:
        while True:
            x,y=left.read(MiB),right.read(MiB)
            if x!=y:equal=False;break
            if not x:break
    if data_identity(a)!=before_a or data_identity(b)!=before_b:
        raise ValueError('capture changed during exact byte comparison')
    return equal

def replace_by_hardlink(source,target,expected_sha):
    """Atomic name replacement, with a temporary rollback hardlink to old bytes."""
    if source.parent.resolve()!=target.parent.resolve():raise ValueError('hardlink replacements stay inside one capture directory')
    if source.is_symlink() or target.is_symlink():raise ValueError('no symlink payloads')
    a,b=identity(source),identity(target)
    if a['device']!=b['device'] or (a['mode'],a['uid'],a['gid'])!=(b['mode'],b['uid'],b['gid']):
        raise ValueError('incompatible filesystem/owner/mode for a hardlink')
    if (a['device'],a['inode'])==(b['device'],b['inode']):return False
    if digest(source)!=expected_sha or digest(target)!=expected_sha or not same_bytes(source,target):
        raise ValueError('candidate bytes differ; no replacement')
    stamp=uuid.uuid4().hex
    staged=target.with_name('.'+target.name+'.dedup-new-'+stamp)
    backup=target.with_name('.'+target.name+'.dedup-old-'+stamp)
    replaced=False
    try:
        os.link(target,backup,follow_symlinks=False)
        os.link(source,staged,follow_symlinks=False)
        if not same_bytes(source,target):raise ValueError('capture changed before atomic replacement')
        os.replace(staged,target);replaced=True
        if digest(target)!=expected_sha:
            os.replace(backup,target);replaced=False
            raise ValueError('post-replacement SHA differs; original target restored')
        backup.unlink() # Only this transaction's temporary rollback link.
        return True
    except BaseException:
        if replaced and backup.exists():os.replace(backup,target)
        raise
    finally:
        for temporary in (staged,backup):
            if temporary.exists():temporary.unlink() # Only temporary links created above.


def _deduplicate_closed(run_root,*,apply=False,process_factory=psutil.Process):
    run,capture,closed=check_closed(run_root,process_factory=process_factory)
    lockpath=run/'.closed-capture-dedup.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entries=collect(capture)
        groups={}
        for entry in entries:
            path=Path(entry['path'])
            if digest(path)!=entry['sha256']:raise ValueError('recorded native SHA mismatch: '+str(path))
            groups.setdefault((entry['before']['device'],entry['size'],entry['sha256']),[]).append(entry)
        report={'schema_version':1,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_at_utc':utc(),'mode':'apply' if apply else 'plan','closed_receipt':closed,
            'run':str(run),'capture':str(capture),'eligible_payload_paths':len(entries),'groups':[],
            'payload_paths_replaced':0,'redundant_logical_bytes_estimate':0,
            'space_scope':'same-size/SHA and exact-byte verified synthetic payloads only; no physical APFS/device reclaimed-byte claim',
            'metadata_scope':'names/sizes/SHA preserved; hardlinked aliases share one inode and its timestamps/metadata; retain captures immutable'}
        journal_path=run/'.closed-capture-dedup-events.jsonl'
        report['journal_path']=str(journal_path) if apply else None
        transaction=uuid.uuid4().hex
        def journal(event):
            if apply:
                with journal_path.open('a') as stream:
                    stream.write(json.dumps({'transaction':transaction,'at_utc':utc(),**event})+'\n');stream.flush()
        for (_,size,sha256),members in sorted(groups.items(),key=lambda item:(item[0][1],item[0][2])):
            if len(members)<2:continue
            distinct={entry['before']['inode'] for entry in members}
            source=Path(members[0]['path']);replaced=[]
            for member in members[1:]:
                target=Path(member['path'])
                if not same_bytes(source,target):raise ValueError('same SHA group failed exact-byte comparison')
                if apply:
                    # The exited PID cannot regain its original birth. Under
                    # the exclusive campaign lock, bracket the closed receipt by
                    # SHA and verify its full stat identity before every link;
                    # never reparse its potentially large sample array per file.
                    if identity(run/'guard.json')!=closed['guard_identity']:
                        raise ValueError('closed guard receipt identity changed')
                    journal({'event':'replacement_requested','source':str(source),'target':str(target),'sha256':sha256,'size':size})
                    changed=replace_by_hardlink(source,target,sha256)
                    if changed:replaced.append(str(target))
                    journal({'event':'replacement_verified','source':str(source),'target':str(target),'sha256':sha256,'size':size,'changed':changed})
            report['redundant_logical_bytes_estimate']+=size*(len(distinct)-1)
            report['payload_paths_replaced']+=len(replaced)
            report['groups'].append({'sha256':sha256,'size':size,'paths':[m['path'] for m in members],
                'distinct_inodes_before':len(distinct),'source':str(source),'replaced':replaced})
        for entry in entries:
            path=Path(entry['path'])
            if path.stat().st_size!=entry['size'] or digest(path)!=entry['sha256']:
                raise ValueError('final payload name/size/SHA verification failed')
        if identity(run/'guard.json')!=closed['guard_identity'] or digest(run/'guard.json')!=closed['guard_sha256']:
            raise ValueError('closed guard receipt changed during consolidation')
        report['closed_guard_identity_and_sha_verified']=True
        report['all_original_names_sizes_sha_verified']=True
        report['finished_at_utc']=utc()
        report['status']='completed'
        return report

def deduplicate(run_root,*,apply=False,process_factory=psutil.Process):
    # Dedup hashing/reads must not overlap a measured research inference.
    with BENCHMARK_LOCK.open('a') as benchmark:
        fcntl.flock(benchmark,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _deduplicate_closed(run_root,apply=apply,process_factory=process_factory)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--apply',action='store_true',help='Actually replace closed duplicate payload names with verified hardlinks; default plans only')
    a=p.parse_args()
    if a.report.exists():raise FileExistsError(a.report)
    try:report=deduplicate(a.run,apply=a.apply)
    except (Exception,KeyboardInterrupt) as error:
        report={'schema_version':1,'status':'failed','error':type(error).__name__+': '+str(error),'run':str(a.run),
            'apply_requested':a.apply,'scope':'no success or space saving claimed; any completed atomic links preserve bytes; inspect per-run transaction journal'}
    a.report.parent.mkdir(parents=True,exist_ok=True);write(a.report,report)
    print(json.dumps({key:report.get(key) for key in ('status','mode','eligible_payload_paths','payload_paths_replaced','redundant_logical_bytes_estimate','all_original_names_sizes_sha_verified','error')}))
    return 0 if report['status']=='completed' else 2
if __name__=='__main__':raise SystemExit(main())
