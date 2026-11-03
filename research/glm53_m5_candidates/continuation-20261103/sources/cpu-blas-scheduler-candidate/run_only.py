"""Execute exactly one reduced CPU/BLAS diagnostic under guarded_build_once.

No compiler, model or GPU is started. The running external guard receipt and its
inherited UID descriptor are mandatory. This file is not a performance runner.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,time
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--guard-receipt',type=Path,required=True)
 p.add_argument('--output',type=Path,default=HERE/'run1');p.add_argument('--receipt',type=Path,default=HERE/'run1-execution.json');a=p.parse_args()
 if a.output.exists() or a.receipt.exists():raise FileExistsError('Fresh native output/receipt required')
 guard=json.loads(a.guard_receipt.read_text());uid=guard['uid_lock'];fd=uid['pass_fd'];opened=os.fstat(fd)
 if guard.get('status')!='running' or (opened.st_dev,opened.st_ino)!=(uid['device'],uid['inode']):raise ValueError('Live external guard and inherited exact UID FD required')
 build=json.loads((HERE/'build-receipt.json').read_text());binary=HERE/'component'
 if build.get('status')!='built_not_executed' or build.get('returncode')!=0 or sha(binary)!=build['binary_sha256']:raise ValueError('Exact successful build required')
 inputs={**build['inputs_sha256'],str(binary):sha(binary),str(HERE/'build-receipt.json'):sha(HERE/'build-receipt.json'),str(Path(__file__).resolve()):sha(__file__)}
 if any(sha(path)!=expected for path,expected in inputs.items()):raise ValueError('Build inputs/binary drift before diagnostic')
 command=[str(binary),str(a.output.resolve())]
 record={'status':'running','scope':'small resident CPU/BLAS scheduler diagnostic, not full M8192 or model causality/performance',
  'started_at_utc':datetime.now(timezone.utc).isoformat(),'command':command,'inputs_sha256':inputs,'build_receipt_sha256':sha(HERE/'build-receipt.json'),
  'model_loaded':False,'gpu_matrix_dispatches':0,'uid_fd_inherited_from_guard':fd,'uid_fd_pass_requested':True}
 try:
  env={k:v for k,v in os.environ.items() if not k.startswith(('AIRLLM_','GGML_','DYLD_','LLAMA_ARG_')) and k!='LLAMA_API_KEY'}
  start=time.perf_counter();r=subprocess.run(command,env=env,pass_fds=(fd,),capture_output=True,text=True,timeout=30)
  record.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,elapsed_seconds=time.perf_counter()-start)
  if r.returncode:raise ValueError('Native diagnostic nonzero; no successful mechanism result')
  data=json.loads(r.stdout);record['measurement']=data
  rows=data['rows'];record['all_expected_backends_match']=all(row['selected_backend_matches_threshold'] for row in rows)
  record['cpu_full_vs_cpu8_all_bitexact']=all(row['cpu_full_vs_cpu8']['cpu_bitexact'] for row in rows)
  record['scheduler_n8_n16_all_cpu8_bitexact']=all(row['scheduled_vs_cpu8']['cpu_bitexact'] for row in rows if row['N']<32)
  record['scheduler_n32_has_bit_differences']=any(not row['scheduled_vs_cpu8']['cpu_bitexact'] for row in rows if row['N']==32)
  for row in rows:
   stem=f"n{row['N']}-seed{row['seed']}"
   checks=(('-cpu8.f32',row['cpu_full_vs_cpu8']['reference_sha256']),('-cpu-full.f32',row['cpu_full_vs_cpu8']['candidate_sha256']),('-scheduled.f32',row['scheduled_vs_cpu8']['candidate_sha256']))
   for suffix,digest in checks:
    path=a.output/(stem+suffix)
    if path.stat().st_size!=row['M']*row['N']*4 or sha(path)!=digest:raise ValueError('Native raw output size/hash differs')
  record['status']='completed_component_diagnostic_only'
 except Exception as error:record.update(status='failed',error=repr(error))
 finally:
  record['inputs_sha256_after']={path:sha(path) for path in inputs};record['inputs_unchanged']=record['inputs_sha256_after']==inputs
  if not record['inputs_unchanged']:record['status']='failed'
  record['finished_at_utc']=datetime.now(timezone.utc).isoformat();a.receipt.write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k not in ('measurement','stdout','stderr','inputs_sha256','inputs_sha256_after')}))
 return 0 if record['status']=='completed_component_diagnostic_only' else 2
if __name__=='__main__':raise SystemExit(main())
