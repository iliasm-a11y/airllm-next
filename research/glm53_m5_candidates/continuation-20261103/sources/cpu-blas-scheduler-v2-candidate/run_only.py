"""One CPU/BLAS fixture with actual-child ownership, inherited UID and GO gate.

Run only under guarded_build_once. No compiler, model or GPU is started here.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,math,os,signal,subprocess,time
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
spec=importlib.util.spec_from_file_location('cpu_blas_owned_native',WORK/'metal-cpu-q8-bridge/native_gpu_run.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(a):
 if a.output.exists() or a.receipt.exists():raise FileExistsError('Fresh native output/receipt required')
 guard=json.loads(a.guard_receipt.read_text());uid=guard['uid_lock'];fd=uid['pass_fd'];opened=os.fstat(fd)
 owner=guard['owned_process'];current=n.psutil.Process()
 if guard.get('status')!='running' or (opened.st_dev,opened.st_ino)!=(uid['device'],uid['inode']) or owner['pid']!=os.getpid() or abs(owner['create_time']-current.create_time())>=.01 or owner['executable']!=current.exe():
  raise ValueError('Live exact supervising guard/owner and inherited UID descriptor required')
 build=json.loads((HERE/'build-receipt.json').read_text());binary=HERE/'component'
 if build.get('status')!='built_not_executed' or build.get('returncode')!=0 or sha(binary)!=build['binary_sha256']:raise ValueError('Exact successful build required')
 inputs={**build['inputs_sha256'],str(binary):sha(binary),str(HERE/'build-receipt.json'):sha(HERE/'build-receipt.json'),str(Path(__file__).resolve()):sha(__file__),str(Path(n.__file__).resolve()):sha(n.__file__)}
 if any(sha(path)!=expected for path,expected in inputs.items()):raise ValueError('Build inputs/binary drift before diagnostic')
 identities={path:n._identity(Path(path)) for path in inputs}
 def intact():
  try:return all(n._identity(Path(path))==expected for path,expected in identities.items())
  except OSError:return False
 limits=n.NativeLimits(**guard['native_limits']);original=n.guard_model(limits);baseline=guard['baseline']
 gate=n.PowerGate('ac');started=time.monotonic();child=awake=identity=None;read_fd=write_fd=None;handlers={}
 record={'status':'starting','scope':'small resident CPU/BLAS scheduler diagnostic; no fullM8192/model causality/performance qualification',
  'started_at_utc':datetime.now(timezone.utc).isoformat(),'inputs_sha256':inputs,'build_receipt_sha256':sha(HERE/'build-receipt.json'),
  'model_loaded':False,'gpu_matrix_dispatches':0,'uid_fd_inherited_from_guard':fd,'uid_lock':uid,'native_limits':vars(limits),
  'component_rss_cap_bytes':128<<20,'max_seconds':30,'samples':[],'baseline':baseline,'guard_violation':None}
 def check(sample,startup=False):
  for key in ('swap_used_bytes','available_bytes'):
   if type(sample.get(key)) is not int or sample[key]<0:raise n.qa.QualificationError('MEMORY_METADATA','Exact memory counters required')
  if 'rss_bytes' in sample and (type(sample['rss_bytes']) is not int or sample['rss_bytes']<0):raise n.qa.QualificationError('MEMORY_METADATA','Exact RSS required')
  code=n.NativeGlmModel._guard_resource_sample(original,sample,baseline,time.monotonic())
  if code:raise n.qa.QualificationError(code,'Original native memory watchdog refused sample')
  pressure=sample.get('pressure_level');growth=max(0,sample['swap_used_bytes']-baseline['swap_used_bytes'])
  if type(pressure) is not int or pressure not in (1,2) or startup and pressure!=1:raise n.qa.QualificationError('MEMORY_ADMISSION','Startup1 and sufficient runtime pressure1/2 required')
  floor=limits.reserve_bytes+(limits.warning_headroom_bytes if pressure==2 else 0)
  if sample['available_bytes']<floor or pressure==2 and growth>limits.warning_swap_growth_bytes:raise n.qa.QualificationError('MEMORY_ADMISSION','Conservative sufficient native headroom required')
  if sample.get('rss_bytes',0)>128<<20:raise n.qa.QualificationError('FIXTURE_RSS_LIMIT','Native fixture128MiB cap')
  if not gate.sample()['passed']:raise n.qa.QualificationError('POWER_CONTEXT_CHANGED','AC/lid/clock context required')
  if not intact():raise n.qa.QualificationError('ARTIFACT_IDENTITY_CHANGED','Owned fixture input changed')
  if time.monotonic()-started>30:raise n.qa.QualificationError('DEADLINE','Fixture30s bound')
  return {**sample,'swap_growth_bytes':growth,'elapsed_seconds':time.monotonic()-started}
 def interrupted(sig,frame):raise n.qa.ProbeInterrupted(sig)
 try:
  for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupted)
  record['admission']=check(n.resource_snapshot(),startup=True)
  env={k:v for k,v in os.environ.items() if not k.startswith(('AIRLLM_','GGML_','DYLD_','LLAMA_ARG_')) and k!='LLAMA_API_KEY'}
  read_fd,write_fd=os.pipe();env['AIRLLM_FIXTURE_START_FD']=str(read_fd)
  command=[str(binary),str(a.output.resolve())];record['command']=command
  with a.receipt.with_suffix('.stdout.log').open('wb') as stdout,a.receipt.with_suffix('.stderr.log').open('wb') as stderr:
   child=subprocess.Popen(command,env=env,pass_fds=(fd,read_fd),stdout=stdout,stderr=stderr)
   os.close(read_fd);read_fd=None
   identity=n.identify(child,binary);record['owned_identity']=identity
   record['uid_fd_observation']=n.observe_uid_fd(child,binary,identity,uid)
   if not record['uid_fd_observation']['observed_matching_lock_fd']:raise n.qa.QualificationError('UID_LOCK_INHERITANCE','Exact UID descriptor not observed in native child')
   record['pre_compute_sample']=check(n.resource_snapshot(child.pid));record['uid_fd_pass_requested']=True
   awake=n.ScopedAwake(child);awake.__enter__();os.write(write_fd,b'G');os.close(write_fd);write_fd=None
   record['go_sent_after_identity_uid_awake_checks']=True;record['status']='running'
   while child.poll() is None:
    try:n.identify(child,binary,identity);awake.check();sample=n.resource_snapshot(child.pid)
    except (n.psutil.NoSuchProcess,RuntimeError,n.qa.QualificationError) as error:
     natural=child.poll() is not None and (isinstance(error,(n.psutil.NoSuchProcess,RuntimeError)) or getattr(error,'code',None)=='OWNED_PROCESS_EXITED')
     if natural:break
     raise
    record['samples'].append(check(sample));time.sleep(.1)
   record['returncode']=child.wait(timeout=5)
  record['postflight_global_sample']=check(n.resource_snapshot())
  stdout_path=a.receipt.with_suffix('.stdout.log');stderr_path=a.receipt.with_suffix('.stderr.log')
  if stdout_path.stat().st_size>2<<20 or stderr_path.stat().st_size>1<<20:raise ValueError('Bounded native logs required')
  record['stdout']=stdout_path.read_text();record['stderr']=stderr_path.read_text()
  if record['returncode']!=0:raise n.qa.QualificationError('NATIVE_NONZERO_EXIT','Native fixture failed')
  data=json.loads(record['stdout']);record['measurement']=data;rows=data['rows']
  record['all_expected_backends_match']=all(row['selected_backend_matches_threshold'] for row in rows)
  record['cpu_full_vs_cpu8_all_bitexact']=all(row['cpu_full_vs_cpu8']['cpu_bitexact'] for row in rows)
  record['scheduler_n8_n16_all_cpu8_bitexact']=all(row['scheduled_vs_cpu8']['cpu_bitexact'] for row in rows if row['N']<32)
  record['scheduler_n32_has_bit_differences']=any(not row['scheduled_vs_cpu8']['cpu_bitexact'] for row in rows if row['N']==32)
  for row in rows:
   stem=f"n{row['N']}-seed{row['seed']}"
   for suffix,digest in (('-cpu8.f32',row['cpu_full_vs_cpu8']['reference_sha256']),('-cpu-full.f32',row['cpu_full_vs_cpu8']['candidate_sha256']),('-scheduled.f32',row['scheduled_vs_cpu8']['candidate_sha256'])):
    path=a.output/(stem+suffix)
    if path.stat().st_size!=row['M']*row['N']*4 or sha(path)!=digest:raise ValueError('Native raw output size/hash differs')
  record['status']='completed_component_diagnostic_only'
 except BaseException as error:record.update(status='cancelled' if isinstance(error,n.qa.ProbeInterrupted) else 'failed',error_code=getattr(error,'code',type(error).__name__),error=str(error))
 finally:
  for sig in handlers:signal.signal(sig,signal.SIG_IGN)
  for descriptor in (read_fd,write_fd):
   if descriptor is not None:os.close(descriptor)
  try:
   if child is not None:record['cleanup']=n.stop_owned(child,binary,identity);record['returncode']=child.poll()
   if awake is not None:awake.close();record['idle_assertion']=awake.record
  except Exception as error:
   record['cleanup_error']=str(error)
   if record['status']=='completed_component_diagnostic_only':record.update(status='failed',error_code='CLEANUP_FAILED')
  gate.sample(final=True);record['power_gate']=gate.summary()
  record['inputs_sha256_after']={path:sha(path) for path in inputs};record['inputs_unchanged']=record['inputs_sha256_after']==inputs
  if record['status']=='completed_component_diagnostic_only' and (not record['inputs_unchanged'] or not record['power_gate']['passed']):record.update(status='failed',error_code='FINAL_EVIDENCE_REJECTED')
  record.update(elapsed_seconds=time.monotonic()-started,finished_at_utc=datetime.now(timezone.utc).isoformat())
  a.receipt.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
  for sig,handler in handlers.items():signal.signal(sig,handler)
 return record

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--guard-receipt',type=Path,required=True)
 p.add_argument('--output',type=Path,default=HERE/'run1');p.add_argument('--receipt',type=Path,default=HERE/'run1-execution.json');a=p.parse_args()
 record=run(a);print(json.dumps({k:record.get(k) for k in ('status','error_code','returncode','all_expected_backends_match','cpu_full_vs_cpu8_all_bitexact','scheduler_n8_n16_all_cpu8_bitexact','scheduler_n32_has_bit_differences')}))
 return 0 if record['status']=='completed_component_diagnostic_only' else 2
if __name__=='__main__':raise SystemExit(main())
