"""Prelaunch UID model ownership for an unchanged native_run.py.

Only this supervisor owns the UID lock; the unchanged wrapper owns its existing
BENCHMARK_LOCK. A small runpy bootstrap passes the held UID lock descriptor to
only the configured native full/restore helper's Popen call. No model is loaded
by the supervisor. The measured wrapper and native source files stay unchanged.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import psutil

REPO=Path(__file__).resolve().parent/'airllm-next'
sys.path[:0]=[str(REPO/'air_llm')]
from airllm.nextgen.glm_native import NativeGlmModel

BOOTSTRAP=r'''
import json,os,runpy,subprocess,sys,time
import psutil
fd=int(sys.argv[1]);events=sys.argv[2];helper=os.path.realpath(sys.argv[3]);wrapper=sys.argv[4];forward=sys.argv[5:]
expected=os.fstat(fd)
def event(value):
    value.update(bootstrap_pid=os.getpid(),wall_time=time.time())
    with open(events,'a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,separators=(',',':'))+'\n');stream.flush()
event({'event':'bootstrap_lock_fd_verified','fd':fd,'device':expected.st_dev,'inode':expected.st_ino})
OriginalPopen=subprocess.Popen
class InheritingPopen(OriginalPopen):
    def __init__(self,*args,**kwargs):
        command=args[0] if args else kwargs.get('args')
        same_helper=(isinstance(command,(list,tuple)) and bool(command) and os.path.realpath(os.fspath(command[0]))==helper)
        selected=(same_helper and len(command)==6 and not kwargs.get('shell',False) and command[-1] in ('full','restore'))
        metadata_only=(same_helper and len(command)==3 and command[1]=='--state-parameters' and not kwargs.get('shell',False))
        if same_helper and not (selected or metadata_only):
            raise RuntimeError('configured native helper command has no admitted descriptor-inheritance contract; no spawn')
        if selected:
            current=os.fstat(fd)
            if (current.st_dev,current.st_ino)!=(expected.st_dev,expected.st_ino):
                raise RuntimeError('UID lock descriptor identity changed before native spawn')
            kwargs['pass_fds']=tuple(sorted(set(kwargs.get('pass_fds',()))|{fd}))
            event({'event':'native_launch_requested','fd':fd,'command':list(command)})
        super().__init__(*args,**kwargs)
        if selected:
            try:birth=psutil.Process(self.pid).create_time()
            except psutil.Error:birth=None
            event({'event':'native_spawned','pid':self.pid,'create_time':birth,'expected_exe':helper,
                   'fd':fd,'pass_fds_requested':True,'device':expected.st_dev,'inode':expected.st_ino})
subprocess.Popen=InheritingPopen
sys.argv=[wrapper]+forward
try:runpy.run_path(wrapper,run_name='__main__')
finally:event({'event':'bootstrap_finished'})
'''


def utc():return datetime.now(timezone.utc).isoformat()
def fingerprint(path):
    p=Path(path);s=p.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.chmod(0o600);temp.replace(path)
def acquire_uid_lock():
    root=Path(tempfile.gettempdir())/('airllm-native-locks-'+str(os.getuid()))
    root.mkdir(mode=0o700,exist_ok=True)
    key=NativeGlmModel._model_lock_key(object.__new__(NativeGlmModel))
    path=root/(key+'.lock');holder=path.open('a')
    try:fcntl.flock(holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as error:
        holder.close();raise RuntimeError('ENGINE_BUSY: UID model lock already held; no wrapper or native helper started') from error
    st=os.fstat(holder.fileno())
    return holder,{'path':str(path),'uid':os.getuid(),'model_key':key,'fd':holder.fileno(),
        'device':st.st_dev,'inode':st.st_ino,'acquired_at_utc':utc(),'acquired_before_wrapper_spawn':True}

def own_identity(pid,exe,previous=None,process_factory=psutil.Process):
    p=process_factory(pid)
    value={'pid':pid,'create_time':p.create_time(),'executable':str(Path(p.exe()).resolve())}
    if not p.is_running() or p.status()==psutil.STATUS_ZOMBIE or value['executable']!=str(Path(exe).resolve()) or previous is not None and value['create_time']!=previous['create_time']:
        raise RuntimeError('owned PID executable or lifetime changed')
    return value

def wait_owned_identity(child,exe,seconds=3):
    # macOS framework Python briefly runs its launcher before re-execing the
    # actual interpreter. Pin birth immediately, then require the final exe.
    started=psutil.Process(child.pid).create_time()
    deadline=time.monotonic()+seconds
    while child.poll() is None and time.monotonic()<deadline:
        process=psutil.Process(child.pid)
        if process.create_time()!=started:raise RuntimeError('owned wrapper PID lifetime changed during startup')
        try:return own_identity(child.pid,exe,{'create_time':started})
        except RuntimeError:time.sleep(.01)
    raise RuntimeError('owned wrapper did not establish the expected executable before startup deadline')


def observe_lock_fd(pid,fd,exe,identity,lock):
    """Read only the one owned helper FD; never enumerate unrelated processes."""
    try:
        own_identity(pid,exe,identity)
        completed=subprocess.run(['/usr/sbin/lsof','-a','-p',str(pid),'-d',str(fd),'-Ffn'],capture_output=True,text=True,timeout=3,check=False)
        names=[line[1:] for line in completed.stdout.splitlines() if line.startswith('n')]
        matched=False
        for name in names:
            try:
                st=Path(name).stat()
                matched=matched or (st.st_dev,st.st_ino)==(lock['device'],lock['inode'])
            except OSError:pass
        own_identity(pid,exe,identity)
        return {'available':True,'observed_matching_lock_fd':matched,'fd':fd,'observed_at_utc':utc(),
            'scope':'OS lsof observation restricted to the one retained helper PID/FD, lifetime and executable bracketed'}
    except (OSError,RuntimeError,psutil.Error,subprocess.SubprocessError) as error:
        return {'available':False,'observed_matching_lock_fd':False,'reason':type(error).__name__+': '+str(error)}

def _events(path):
    if not path.exists():return []
    rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:continue
    return rows

def supervise(wrapper,forward,receipt,*,poll_seconds=.25):
    if not 0<poll_seconds<=.5:raise ValueError('ownership polling must be<=0.5s')
    wrapper=Path(wrapper).resolve();receipt=Path(receipt).resolve()
    args=list(forward)
    if args and args[0]=='--':args=args[1:]
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('full','restore'),required=True)
    target,_=p.parse_known_args(args)
    helper=target.binary.resolve();output=target.output.resolve()
    if receipt.is_relative_to(output):raise ValueError('supervisor receipt must be outside the wrapper output directory')
    if not wrapper.is_file() or not helper.is_file():raise ValueError('existing wrapper and helper files required')
    if receipt.exists():raise FileExistsError(receipt)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    events=receipt.with_suffix(receipt.suffix+'.events.jsonl')
    with events.open('x') as stream:os.fchmod(stream.fileno(),0o600)
    original={'wrapper':fingerprint(wrapper),'helper':fingerprint(helper)}
    own=psutil.Process(os.getpid())
    interpreter_executable=Path(own.exe()).resolve()
    report={'schema_version':1,'status':'starting','started_at_utc':utc(),
        'supervisor':{'pid':os.getpid(),'create_time':own.create_time(),'executable':own.exe()},
        'supervisor_sha256':sha(__file__),'bootstrap_sha256':hashlib.sha256(BOOTSTRAP.encode()).hexdigest(),
        'wrapper':{'path':str(wrapper),'sha256':sha(wrapper),'file_identity':original['wrapper']},
        'helper':{'path':str(helper),'sha256':sha(helper),'file_identity':original['helper']},
        'wrapper_arguments':args,'events_path':str(events),'mode':target.mode,'output':str(output),
        'benchmark_lock_policy':'supervisor does not acquire BENCHMARK_LOCK; unchanged wrapper acquires its existing nonblocking benchmark lock',
        'native_children':[],'native_signal_actions':[],'wrapper_signal_actions':[],
        'scope':'prelaunch UID model ownership and descriptor inheritance only; resource/power/numerical qualification remains owned by the unchanged wrapper',
        'cleanup':{'parent_uid_fd_closed':False}}
    holder=child=identity=None;handlers={};stop=[None];notified=False
    def interrupted(signum,unused):stop[0]=signum
    try:
        holder,lock=acquire_uid_lock();report['uid_model_lock']=lock
        write(receipt,report)
        for signum in (signal.SIGINT,signal.SIGTERM):handlers[signum]=signal.signal(signum,interrupted)
        command=[sys.executable,'-c',BOOTSTRAP,str(holder.fileno()),str(events),str(helper),str(wrapper),*args]
        with receipt.with_suffix(receipt.suffix+'.wrapper-stdout.log').open('wb') as stdout,receipt.with_suffix(receipt.suffix+'.wrapper-stderr.log').open('wb') as stderr:
            child=subprocess.Popen(command,pass_fds=(holder.fileno(),),stdout=stdout,stderr=stderr)
            # Process executable stays Python because runpy runs the unchanged wrapper in place.
            identity=wait_owned_identity(child,interpreter_executable)
            report.update(status='running',wrapper_process=identity,wrapper_started_at_utc=utc())
            print(json.dumps({'status':'running','supervisor_pid':os.getpid(),'wrapper_pid':child.pid,'receipt':str(receipt)}),flush=True)
            seen={}
            while True:
                if fingerprint(wrapper)!=original['wrapper'] or fingerprint(helper)!=original['helper']:
                    report.setdefault('ownership_violation','wrapper/helper source or binary identity changed')
                    stop[0]=signal.SIGTERM
                for event in _events(events):
                    if event.get('event')=='native_spawned' and event.get('pid') not in seen:
                        seen[event['pid']]={**event,'identity_verified':False,'fd_observation':None}
                alive=[]
                for pid,row in seen.items():
                    try:
                        candidate=own_identity(pid,helper)
                        if row.get('create_time') is None or candidate['create_time']!=row['create_time']:
                            raise RuntimeError('native helper birth not recorded or changed')
                        row['identity_verified']=True;row['verified_identity']=candidate;row['last_alive_at_utc']=utc()
                        if row.get('fd_observation') is None or not row['fd_observation'].get('observed_matching_lock_fd'):
                            row['fd_observation']=observe_lock_fd(pid,row['fd'],helper,candidate,lock)
                        alive.append(pid)
                    except psutil.NoSuchProcess:
                        row['exited_or_gone_at_utc']=utc()
                    except (psutil.Error,RuntimeError) as error:
                        # A zombie has exited. Any other identity change is recorded;
                        # never signal the PID that failed identity verification.
                        row['identity_or_exit_observation']=type(error).__name__+': '+str(error)
                report['native_children']=list(seen.values())
                report['last_observed_at_utc']=utc()
                if child.poll() is None:
                    own_identity(child.pid,interpreter_executable,identity)
                    if stop[0] is not None and not notified:
                        # The wrapper's existing handler owns native cancellation/cleanup.
                        child.send_signal(signal.SIGTERM)
                        report['wrapper_signal_actions'].append({'signal':'SIGTERM','at_utc':utc(),'verified_identity':identity})
                        notified=True
                elif not alive:
                    break
                else:
                    report['status']='holding_uid_lock_after_wrapper_exit'
                write(receipt,report)
                time.sleep(poll_seconds)
            report['wrapper_returncode']=child.wait(timeout=5)
            report['status']='completed' if report['wrapper_returncode']==0 and bool(report['native_children']) and all(row.get('identity_verified') and row.get('pass_fds_requested') for row in report['native_children']) and not report.get('ownership_violation') and stop[0] is None else 'failed'
    except Exception as error:
        report.update(status='blocked' if child is None else 'failed',error=type(error).__name__+': '+str(error))
    finally:
        for signum in handlers:signal.signal(signum,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            try:
                own_identity(child.pid,interpreter_executable,identity)
                child.send_signal(signal.SIGTERM)
                report['wrapper_signal_actions'].append({'signal':'SIGTERM','at_utc':utc(),'reason':'supervisor error cleanup'})
                child.wait(timeout=10)
            except Exception as error:
                report['cleanup']['wrapper_cleanup_error']=str(error)
                # No direct native-helper signal. Its inherited FD retains the UID lock.
        if child is not None:report['wrapper_returncode']=child.poll()
        if holder is not None:
            holder.close();report['cleanup'].update(parent_uid_fd_closed=True,closed_at_utc=utc())
        report['cleanup']['native_helpers_signalled']=False
        report['cleanup']['surviving_native_helper_policy']='each launched native helper inherited the UID FD; a surviving helper retains model ownership after parent closure'
        report['finished_at_utc']=utc()
        report['wrapper_file_unchanged']=fingerprint(wrapper)==original['wrapper']
        report['helper_file_unchanged']=fingerprint(helper)==original['helper']
        write(receipt,report)
        for signum,handler in handlers.items():signal.signal(signum,handler)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wrapper',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('wrapper_arguments',nargs=argparse.REMAINDER)
    a=p.parse_args()
    report=supervise(a.wrapper,a.wrapper_arguments,a.receipt)
    print(json.dumps({'status':report['status'],'receipt':str(a.receipt),'wrapper_returncode':report.get('wrapper_returncode')}))
    return 0 if report['status']=='completed' else 2
if __name__=='__main__':raise SystemExit(main())
