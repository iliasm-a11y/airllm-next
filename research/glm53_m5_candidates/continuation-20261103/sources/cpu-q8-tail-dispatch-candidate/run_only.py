"""One direct CPU Q8 tail fixture with actual-child ownership, inherited UID and GO gate.

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
 record={'status':'starting','scope':'resident Q8 M8192/K4096 tail dispatch only; no GLM/fences/performance qualification',
  'started_at_utc':datetime.now(timezone.utc).isoformat(),'inputs_sha256':inputs,'build_receipt_sha256':sha(HERE/'build-receipt.json'),
  'model_loaded':False,'gpu_matrix_dispatches':0,'uid_fd_inherited_from_guard':fd,'uid_lock':uid,'native_limits':vars(limits),
  'component_rss_cap_bytes':128<<20,'max_seconds':120,'samples':[],'baseline':baseline,'guard_violation':None}
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
  if time.monotonic()-started>120:raise n.qa.QualificationError('DEADLINE','Fixture120s bound')
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
  expected_n={14,18,20,22,24,26,28,30,32,38,44,48,64,72,80,88,96,104,112,120,128}
  expected_pairs={(n,seed) for n in expected_n for seed in (5305,713962)}
  if len(rows)!=42 or {(row['N'],row['seed']) for row in rows}!=expected_pairs:raise ValueError('Exact42-case coverage required')
  if data.get('status')!='completed_q8_tail_dispatch_diagnostic_only' or data.get('cpu_threads')!=10 or data.get('uses_reference_mode') is not False or data.get('BLAS_initialized') is not False or data.get('scheduler_used') is not False or data.get('gpu_matrix_dispatches')!=0 or data.get('raw_float32_exported') is not False:raise ValueError('Native CPU-only scope differs')
  if type(data.get('native_lifetime_peak_rss_bytes')) is not int or not 0<data['native_lifetime_peak_rss_bytes']<=128<<20:raise ValueError('Native lifetime128MiB evidence required')
  result=a.output/'result.json'
  if json.loads(result.read_text())!=data or sorted(p.name for p in a.output.iterdir())!=['result.json']:raise ValueError('Native result differs or unexpected raw export')
  record['native_result_sha256']=sha(result);record['raw_float32_exported']=False
  record['raw_scope']='Native in-memory comparisons/SHA only; no independent F32 export validation'
  for row in rows:
   if row.get('K')!=4096 or row.get('M')!=8192 or row.get('actual_compute_backend')!='CPU' or row.get('native_vec_dot_nrows')!=2 or row.get('raw_float32_exported') is not False:raise ValueError('Native shape/backend/real nrc2 traits required')
   if sum(row['reference8_parts'])!=row['N'] or sum(row['v4_parts'])!=row['N']:raise ValueError('Partition coverage differs')
   for key in ('fullN_vs_reference8','v4_vs_reference8'):
    value=row[key]
    if value.get('outputs')!=8192*row['N'] or value.get('all_finite') is not True or type(value.get('different_float32_bits')) is not int or not 0<=value['different_float32_bits']<=value['outputs'] or value.get('cpu_bitexact')!=(value['different_float32_bits']==0) or type(value.get('max_abs_error')) not in (int,float) or not math.isfinite(value['max_abs_error']) or value['max_abs_error']<0:raise ValueError('Invalid native comparison')
    for field in ('reference_sha256','candidate_sha256'):
     digest=value[field]
     if type(digest) is not str or len(digest)!=64 or set(digest)-set('0123456789abcdef'):raise ValueError('Exact output SHA required')
   if row['fullN_vs_reference8']['reference_sha256']!=row['v4_vs_reference8']['reference_sha256']:raise ValueError('Reference identity differs')
  record['fullN_vs_reference8_all_bitexact']=all(row['fullN_vs_reference8']['cpu_bitexact'] for row in rows)
  record['fullN_different_cases']=[{'N':row['N'],'seed':row['seed'],'different_float32_bits':row['fullN_vs_reference8']['different_float32_bits'],'max_abs_error':row['fullN_vs_reference8']['max_abs_error']} for row in rows if not row['fullN_vs_reference8']['cpu_bitexact']]
  record['v4_vs_reference8_all_bitexact']=all(row['v4_vs_reference8']['cpu_bitexact'] for row in rows)
  record['v4_shape_gate_passed']=record['v4_vs_reference8_all_bitexact']
  record['v4_model_fidelity_qualified']=False
  if not record['v4_shape_gate_passed']:raise n.qa.QualificationError('Q8_V4_DIFFERENCE','The proposed v4 partition differs from CPU8; preserve results and reject before model')
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
 record=run(a);print(json.dumps({k:record.get(k) for k in ('status','error_code','returncode','fullN_vs_reference8_all_bitexact','v4_vs_reference8_all_bitexact','v4_shape_gate_passed')}))
 return 0 if record['status']=='completed_component_diagnostic_only' else 2
if __name__=='__main__':raise SystemExit(main())
