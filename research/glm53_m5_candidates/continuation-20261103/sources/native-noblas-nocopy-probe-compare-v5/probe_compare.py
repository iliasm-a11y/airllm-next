"""v5: strict closed GPU128 comparison with exact producer KDA alias checks.

This command never builds/loads a model. A matching window is never a complete
prompt, full arm, snapshot, performance or EOS qualification. Existing complete
comparators are neither changed nor called with rewritten completion metadata.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
BASE=WORK/'native-noblas-nocopy-candidate'
BUILD_SUPERVISOR=WORK/'native-nocopy-build-supervisor-v2'
REPO=WORK/'airllm-next'
PREPARE=HERE/'probe-compare-prepare.json'
FROZEN=WORK/'native-prompts-max/native-prompts.json'
REFERENCE_BUILD=WORK/'native-prefill-final-controls.build.json'
REFERENCE_WRAPPER=WORK/'publication/research/glm53_batched_prefill/native_run.py'
REFERENCE_MANIFEST=Path('/Users/macbook/Documents/Codex/2026-11-02/referenced-chatgpt-conversation-this-is-an-2/work/runtime-v4/manifest.json')
REFERENCE_GUARD={'path':str(WORK/'native-full8/guard.json'),'sha256':'b991594498b14afc2a8b273da87843fb78e5534a023a053df393e2b6be0ecd08','bytes':26277130}
REFERENCE_GUARD_READ_LIMIT=32<<20
KDA_ALIASES={f'kda_scan_out-{layer} (reshaped)':f'kda_scan_out-{layer}'for layer in range(45)if layer%4!=3}
DELEGATES={
    'native':('research.glm53_batched_prefill.native_compare',REPO/'research/glm53_batched_prefill/native_compare.py'),
    'batch':('research.glm53_batched_prefill.batch_compare',REPO/'research/glm53_batched_prefill/batch_compare.py'),
    'fidelity':('research.glm53_m5.compare_fidelity',REPO/'research/glm53_m5/compare_fidelity.py'),
    'key':('research.glm53_batched_prefill.snapshot_key',REPO/'research/glm53_batched_prefill/snapshot_key.py'),
    'native_guard':('air_llm.airllm.nextgen.glm_native',REPO/'air_llm/airllm/nextgen/glm_native.py'),
    'power':('research.glm53_gpu_v3.power_state',REPO/'research/glm53_gpu_v3/power_state.py'),
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read(path,limit=16<<20):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>limit:raise ValueError('bounded existing JSON required: '+str(path))
    return json.loads(path.read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON: '+v)))

def descriptor(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}

def checked_file(row):
    if type(row) is not dict or set(row)!={'path','sha256','bytes'} or type(row['path']) is not str or not Path(row['path']).is_absolute():raise ValueError('complete absolute file descriptor required')
    path=Path(row['path'])
    if type(row['bytes']) is not int or row['bytes']<1 or not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('actual bound file differs: '+str(path))
    return path

def identity(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def pinned_reference_guard(path,prepared,*,parse=True):
    # Only this exact immutable historical receipt gets the larger bound. The
    # candidate and every other JSON still use read()'s original16MiB default.
    if prepared.get('reference_guard')!=REFERENCE_GUARD or prepared.get('reference_guard_read_limit_bytes')!=REFERENCE_GUARD_READ_LIMIT:raise ValueError('exact historical CPU8 guard provenance required')
    path=Path(path).resolve()
    if str(path)!=REFERENCE_GUARD['path']:raise ValueError('larger JSON read refused outside exact historical CPU8 guard path')
    before=identity(path)
    if before[2]!=REFERENCE_GUARD['bytes']or before[2]>REFERENCE_GUARD_READ_LIMIT:raise ValueError('historical CPU8 guard size differs or exceeds32MiB')
    with path.open('rb')as source:payload=source.read(REFERENCE_GUARD_READ_LIMIT+1)
    if len(payload)!=REFERENCE_GUARD['bytes']or hashlib.sha256(payload).hexdigest()!=REFERENCE_GUARD['sha256']or identity(path)!=before:raise ValueError('historical CPU8 guard bytes/SHA/identity differ')
    # The hash is over precisely the bounded bytes passed to the JSON parser.
    return json.loads(payload,parse_constant=lambda v:(_ for _ in()).throw(ValueError('nonfinite JSON: '+v)))if parse else dict(REFERENCE_GUARD)

def verify_sources(prepared):
    if prepared.get('status')!='source_only_review_required':raise ValueError('explicit source preparation required')
    if prepared.get('reference_guard')!=REFERENCE_GUARD or prepared.get('reference_guard_read_limit_bytes')!=REFERENCE_GUARD_READ_LIMIT:raise ValueError('explicit immutable CPU8 guard read exception required')
    required={str(path.resolve()) for _,path in DELEGATES.values()}
    required|={str((BASE/name).resolve()) for name in ('probe_guard.py','gpu_contract.py','profile.py','native_gpu_run.py','build_driver.py','driver/research/glm53_batched_prefill/native_prefill.cpp')}
    required|={str(path.resolve()) for path in (HERE/'probe_compare.py',HERE/'build_guard.py',BUILD_SUPERVISOR/'supervised_build.py',BUILD_SUPERVISOR/'source-manifest.json')}
    required.add(str((WORK/'native-gpu-two-build-compare/provenance.py').resolve()))
    pins=prepared.get('source_pins',{})
    if not required<=set(pins):raise ValueError('complete comparison/runtime/delegate pins required')
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('pinned source differs: '+path)
    for name,path in DELEGATES.values():
        module=sys.modules.get(name)
        if module is not None and Path(getattr(module,'__file__','')).resolve()!=path.resolve():raise ValueError('foreign cached delegate: '+name)
    # The guard is imported under both public package spellings by existing code.
    for name in ('airllm.nextgen.glm_native','air_llm.airllm.nextgen.glm_native'):
        module=sys.modules.get(name)
        if module is not None and Path(getattr(module,'__file__','')).resolve()!=DELEGATES['native_guard'][1].resolve():raise ValueError('foreign cached native guard')
    for name in ('gpu_contract','profile','probe_guard'):
        module=sys.modules.get(name)
        if module is not None and Path(getattr(module,'__file__','')).resolve()!=(BASE/(name+'.py')).resolve():raise ValueError('foreign cached sibling: '+name)

def load_exact(tag,path):
    payload=Path(path).read_bytes();name=tag+'_'+hashlib.sha256(payload).hexdigest()
    previous=sys.modules.get(name)
    if previous is not None:
        if Path(getattr(previous,'__file__','')).resolve()!=Path(path).resolve():raise ValueError('foreign cached hashed module')
        return previous
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:exec(compile(payload,str(path),'exec'),module.__dict__)
    except BaseException:sys.modules.pop(name,None);raise
    if Path(path).read_bytes()!=payload:raise ValueError('module changed while loading')
    return module

def modules(prepared):
    verify_sources(prepared);sys.path[:0]=[str(REPO),str(REPO/'air_llm')]
    answer={label:importlib.import_module(name) for label,(name,path) in DELEGATES.items()}
    answer['gpu']=load_exact('noblas_nocopy_probe_contract',BASE/'gpu_contract.py')
    answer['probe_guard']=load_exact('noblas_nocopy_probe_guard',BASE/'probe_guard.py')
    answer['build_guard']=load_exact('noblas_nocopy_dedicated_build_guard_v3',HERE/'build_guard.py')
    # Only verify_build(reference=True) is reused; no old GPU/tail policy is delegated.
    answer['reference_provenance']=load_exact('probe_reference_build_provenance',WORK/'native-gpu-two-build-compare/provenance.py')
    verify_sources(prepared);return answer

def unique_input(build,suffix):
    rows=[(Path(path),value) for path,value in build['inputs_sha256'].items() if path.endswith('/'+suffix)]
    if len(rows)!=1 or sha(rows[0][0])!=rows[0][1]:raise ValueError('exact unique build input required: '+suffix)
    return rows[0][0]

def role_input(build,role):
    path=build.get('input_roles',{}).get(role)
    if type(path)is not str or not Path(path).is_absolute():raise ValueError('explicit absolute build input role required: '+role)
    digest=build.get('inputs_sha256',{}).get(path)
    if type(digest)is not str or re.fullmatch('[0-9a-f]{64}',digest)is None or sha(path)!=digest:raise ValueError('actual hashed input role differs: '+role)
    return Path(path)

def actual_arm(build_path,manifest_path,wrapper_path,*,gpu=False):
    build=read(build_path);command=build['command']
    if type(command) is not list or command.count('-o')!=1:raise ValueError('actual compile command required')
    binary=Path(command[command.index('-o')+1])
    # The new build tracks both CPU-v4 geometry and the real GPU helper. A
    # filename suffix is ambiguous; the explicit hashed semantic role is exact.
    if gpu and (build.get('input_roles',{}).get('helper_source')!=str(BASE/'driver/research/glm53_batched_prefill/native_prefill.cpp')or binary!=Path(build_path).resolve().parent/'native-prefill-noblas-nocopy'):raise ValueError('exact reviewed helper and isolated output path required before hashing')
    helper=role_input(build,'helper_source')if gpu else unique_input(build,'glm53_batched_prefill/native_prefill.cpp')
    manifest=read(manifest_path);directory=Path(manifest_path).parent/'bin'
    libraries={Path(row['path']).name:descriptor(Path(manifest_path).parent/row['path']) for row in manifest['artifacts'] if row['path'].endswith('.dylib') and 'symlink' not in row}
    if not libraries:raise ValueError('complete actual library package required')
    paths={'helper_source':helper,'snapshot_source':helper.with_name('snapshot.h'),'state_capture_source':helper.parent.parent/'glm53_gpu_v3/state_capture.inc','native_manifest':Path(manifest_path),'wrapper_source':Path(wrapper_path)}
    arm={'build_receipt':descriptor(build_path),'binary':descriptor(binary),'helper_source':descriptor(helper),'runtime_manifest':descriptor(manifest_path),
         'libraries':libraries,'runtime_binaries':{'helper':sha(binary),**{name:row['sha256'] for name,row in libraries.items()}},
         'source_sha256':{name:sha(path) for name,path in paths.items()},'source_files':{name:descriptor(path) for name,path in paths.items()},'libraries_directory':str(directory.resolve())}
    if gpu:
        if build.get('status')!='built_metadata_only_smoked' or build.get('binary_sha256')!=sha(binary):raise ValueError('real completed new GPU driver build required')
    return arm,build

def make_binding(prepared,m,driver_path,provider_path,profile_path):
    verify_sources(prepared)
    if sha(provider_path)!=prepared['provider_build_receipt_sha256']:raise ValueError('exact closed NoCopy provider receipt required')
    a,ab=actual_arm(REFERENCE_BUILD,REFERENCE_MANIFEST,REFERENCE_WRAPPER)
    m['reference_provenance'].verify_build(a,reference=True)
    if a['source_sha256']['wrapper_source']!=prepared['reference_wrapper_sha256']:raise ValueError('original measured CPU8 wrapper differs')
    driver=read(driver_path);candidate_manifest=Path(driver['input_roles']['runtime_manifest'])
    if candidate_manifest.resolve()!=Path(prepared['candidate_runtime_manifest']).resolve():raise ValueError('exact existing NoCopy runtime package required')
    b,bb=actual_arm(driver_path,candidate_manifest,BASE/'native_gpu_run.py',gpu=True)
    guarded_build=m['build_guard'].validate(bb,Path(driver_path),prepared,m)
    provider=read(provider_path);profile=m['gpu'].validate_profile(read(profile_path))
    m['gpu'].validate_artifacts(profile,provider,bb,libraries=b['libraries_directory'],binary=b['binary']['path'],helper_source=b['helper_source']['path'],runner_source=BASE/'native_gpu_run.py',guard_source=DELEGATES['native_guard'][1],provider_receipt_sha256=sha(provider_path))
    if b['helper_source']['sha256']!=prepared['candidate_helper_sha256']:raise ValueError('candidate differs from reviewed source')
    for name in ('snapshot_source','state_capture_source'):
        if a['source_sha256'][name]!=b['source_sha256'][name]:raise ValueError('CPU/GPU capture dependency changed')
    # Keep the exact runner's mathematical source map. Build supervision is an
    # additional bound evidence envelope below, not a silent runtime-key change.
    extra={'gpu_contract_source':BASE/'gpu_contract.py','gpu_profile_source':BASE/'profile.py','gpu_profile_document':Path(profile_path),'provider_build_receipt':Path(provider_path),'gpu_driver_build_receipt':Path(driver_path),'native_guard_source':DELEGATES['native_guard'][1]}
    b['source_sha256'].update({name:sha(path) for name,path in extra.items()})
    b['source_files'].update({name:descriptor(path) for name,path in extra.items()})
    reference_guard=pinned_reference_guard(REFERENCE_GUARD['path'],prepared,parse=False)
    verify_sources(prepared)
    return {'schema_version':1,'status':'actual_two_build_probe128_binding','preparation_sha256':sha(PREPARE),
        'reference':a,'candidate':b,'provider_build_receipt':descriptor(provider_path),'gpu_profile':descriptor(profile_path),'driver_build_supervision':guarded_build,'reference_guard':reference_guard,
        'frozen_prompts':descriptor(FROZEN),'frozen_prompts_expected_sha256':m['gpu'].FROZEN_PROMPTS_SHA256,
        'full_qualification':False,'model_numeric_qualified':False,'scope':'actual builds only; no model execution or numerical qualification'}

def load_binding(path,prepared,m):
    binding=read(path)
    if binding.get('status')!='actual_two_build_probe128_binding':raise ValueError('actual two-build binding required; pending/null disallowed')
    for arm in (binding['reference'],binding['candidate']):
        for field in ('build_receipt','binary','helper_source','runtime_manifest'):checked_file(arm[field])
        for row in arm['libraries'].values():checked_file(row)
        for row in arm['source_files'].values():checked_file(row)
    expected=make_binding(prepared,m,checked_file(binding['candidate']['build_receipt']),checked_file(binding['provider_build_receipt']),checked_file(binding['gpu_profile']))
    if binding!=expected:raise ValueError('binding differs from the two actual revalidated builds/profile/source pins')
    checked_file(binding['frozen_prompts'])
    return binding,read(binding['reference']['build_receipt']['path']),read(binding['candidate']['build_receipt']['path']),read(binding['gpu_profile']['path'])

def closed_guard(receipt,*,gpu,m):
    if gpu:m['probe_guard'].validate_probe_guard(receipt)
    else:m['native']._guard(receipt)
    if not m['power'].power_evidence_matches(receipt,'ac'):raise ValueError('raw closed AC/idle/continuity evidence differs')
    cleanup=receipt.get('cleanup',{})
    if cleanup.get('returncode')!=0 or cleanup.get('owned_server_stopped') is not True or cleanup.get('signals')!=[]:raise ValueError('natural owned completion required')
    # The historical CPU8 receipt predates this additional explicit field. Its
    # original integrity_passed + tracked wrapper identity and bound wrapper SHA
    # remain required; do not invent a retrospective field in that receipt.
    if gpu and receipt.get('runner_source_identity_unchanged') is not True:raise ValueError('runner postflight source identity missing')
    for path,value in receipt['preflight']['tracked_identities'].items():
        if identity(path)!=value:raise ValueError('tracked measured input changed: '+path)
    if gpu:
        uid=receipt.get('uid_model_lock',{});gate=receipt.get('start_gate',{})
        if uid.get('acquired_before_spawn') is not True or uid.get('pass_fds_requested') is not True or uid.get('fd_observation',{}).get('observed_matching_lock_fd') is not True or uid.get('owned_child_pid')!=receipt.get('owned_pid') or uid.get('child_poll_before_parent_fd_close')!=0 or uid.get('parent_fd_closed') is not True or gate.get('released') is not True or gate.get('uid_fd_observed_before_GO') is not True or gate.get('power_verified_before_GO') is not True or gate.get('owned_identity_before_GO')!=receipt.get('owned_identity'):raise ValueError('closed inherited UID/owned GO evidence required')

def dependencies(root,guard_path,report,receipt,arm,build,m,original,profile=None):
    manifest=receipt['preflight']['snapshot_manifest'];key,_=m['key'].canonical_key(manifest)
    if key!=report.get('snapshot_key_sha256') or key!=receipt['preflight'].get('snapshot_key_sha256'):raise ValueError('own complete manifest/key differs')
    sources=arm['source_sha256'];compound=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if receipt['preflight'].get('source_sha256')!=sources or manifest['runtime_source_sha256']!=compound or manifest['runtime_binaries']!=arm['runtime_binaries'] or receipt['preflight'].get('runtime_binaries')!=arm['runtime_binaries']:raise ValueError('captured runtime/source hashes differ from own actual build')
    params=manifest['state_parameters'];ub=report.get('microbatch');command=receipt.get('command');prompt_path=Path(guard_path).parent/'run-prompts.json'
    expected=[str(Path(arm['binary']['path']).resolve()),None,str(prompt_path.resolve()),str(root.resolve()),str(ub),'probe128' if profile else 'full']
    if type(command) is not list or len(command)!=6 or any(command[i]!=expected[i] for i in (0,2,3,4,5)):raise ValueError('owned helper/input/capture command differs')
    if command[1] not in receipt['preflight']['tracked_identities']:raise ValueError('actual first model shard is outside tracked inputs')
    transmitted=read(prompt_path);wanted=copy.deepcopy(original);wanted.update(snapshot_key_sha256=key,native_snapshot_scratch_bound_bytes=16<<20)
    if transmitted!=wanted:raise ValueError('transmitted input differs from all600 frozen prompts plus the two explicit envelope fields')
    if report.get('state_parameters')!=params or report.get('logical_batch')!=ub or params.get('n_batch')!=ub or params.get('n_ubatch')!=ub or params.get('n_threads')!=10 or params.get('n_threads_batch')!=10:raise ValueError('actual keyed math parameters differ')
    if manifest['prefix_tokens']!=original['agents'][0]['token_ids'][:208] or manifest['prefix_positions']!=list(range(208)):raise ValueError('own key prefix differs from original native IDs')
    checkpoint=receipt['preflight']['metadata']['checkpoint']
    if checkpoint.get('integrity_verified') is not True or checkpoint.get('errors'):raise ValueError('original model integrity metadata required')
    norm=copy.deepcopy(manifest)
    if profile is None:
        if report.get('complete') is not True or report.get('mode')!='full' or report.get('diagnostic') is not True or len(report.get('requests',[]))!=3 or [r.get('agent_id') for r in report['requests']]!=[1,600,1] or ub!=8 or build['parameter_smoke']['8']!=params:raise ValueError('closed original complete CPU8 reference required')
    else:
        m['gpu'].validate_capture_report(report,receipt['preflight'],profile,arm['libraries_directory'],receipt['baseline']['swap_used_bytes'],ub)
        if receipt['preflight'].get('original_prompts_sha256')!=m['gpu'].FROZEN_PROMPTS_SHA256:raise ValueError('candidate frozen input byte identity missing')
        agents_sha=hashlib.sha256(json.dumps(original['agents'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if receipt['preflight'].get('original_agents_sha256')!=agents_sha:raise ValueError('candidate all600-agent identity missing')
        smoke=build['parameter_smoke'][str(ub)]
        if smoke.get('returncode')!=0 or smoke.get('state_parameters')!=params:raise ValueError('candidate actual build smoke parameters differ')
        for field in ('prefill_chunk_policy','prefill_chunk_policy_version','tail_reference_microbatch','prefill_fences','fallback_column_chunk','fallback_remainder_policy','cpu_accelerator_policy','diagnostic_scope','execution_backend','cpu_metal_q8_bridge_compiled','gpu_execution_possible','gpu_bridge_requested','gpu_policy_sha256','weight_storage_policy','permanent_gpu_weight_bytes'):del norm['state_parameters'][field]
        for field,value in profile['render_options_extension'].items():
            if norm['render_options'].get(field)!=value:raise ValueError('own keyed GPU profile differs')
            del norm['render_options'][field]
        for field,value in profile['environment'].items():
            if norm['render_options']['native_environment'].get(field)!=value:raise ValueError('own GPU environment differs')
            del norm['render_options']['native_environment'][field]
        window={'request_index':0,'agent_id':1,'token_first':0,'token_count':128,'generation':False,'full_campaign':False}
        if norm['render_options'].pop('diagnostic_window',None)!=window:raise ValueError('keyed partial window differs')
    for name in ('n_batch','n_ubatch'):del norm['state_parameters'][name]
    norm['runtime_source_sha256']='separately-verified-two-real-builds';norm['runtime_binaries']='separately-verified-two-real-builds'
    return norm,key,command[1]

def candidate_alias_catalog(rows,wanted,names):
    """Pure header gate: exact118 canonical +34 known reshape names per batch."""
    if len(names)!=118 or len(KDA_ALIASES)!=34 or set(KDA_ALIASES.values())!={name for name in names if name.startswith('kda_scan_out-')}:raise ValueError('fixed canonical/KDA alias catalog differs')
    allowed=set(names)|set(KDA_ALIASES);catalog={};files=set()
    fields={'request','phase','name','offset','input_tokens','batch_index','type','shape','strides','bytes','file','sha256'}
    expected={(name,*interval)for name in allowed for interval in wanted}
    for row in rows:
        if type(row)is not dict or set(row)!=fields:raise ValueError('exact native tensor header fields required')
        if any(type(row.get(field))is not int for field in('request','offset','input_tokens','batch_index','type','bytes')):raise ValueError('exact integer candidate tensor metadata required')
        if row['request']!=0 or row['phase']!='probe128'or type(row['name'])is not str or row['name']not in allowed:raise ValueError('candidate tensor name/request/phase outside exact canonical+KDA alias catalog')
        key=(row['name'],row['offset'],row['input_tokens'],row['batch_index'])
        if key not in expected or key in catalog:raise ValueError('duplicate/out-of-window candidate tensor header')
        if any(type(row[field])is not list or len(row[field])!=4 or any(type(value)is not int or value<(1 if field=='shape'else 0)for value in row[field])for field in('shape','strides')):raise ValueError('exact shape/stride integers required')
        if not 0<row['bytes']<=64<<20 or type(row['file'])is not str or re.fullmatch(r'tensor-[0-9]+\.bin',row['file'])is None or row['file']in files or type(row['sha256'])is not str or re.fullmatch('[0-9a-f]{64}',row['sha256'])is None:raise ValueError('distinct native dump file/size/SHA metadata required')
        catalog[key]=row;files.add(row['file'])
    if set(catalog)!=expected:raise ValueError('candidate requires exactly118 canonical +34 known KDA aliases per batch')
    pairs=[]
    for alias,base in sorted(KDA_ALIASES.items()):
        for interval in wanted:
            a=catalog[(alias,*interval)];b=catalog[(base,*interval)];n=interval[1]
            if b['type']!=0 or b['shape']!=[128,64,n,1]or b['strides']!=[4,512,32768,32768*n]or b['bytes']!=32768*n:raise ValueError('canonical KDA layout differs from pinned contiguous native producer')
            if any(a[field]!=b[field]for field in fields-{'name','file'}):raise ValueError('known KDA alias header differs from same-batch canonical tensor')
            pairs.append((a,b))
    canonical=[row for key,row in catalog.items()if key[0]in names]
    return canonical,pairs

def verify_alias_payloads(root,pairs,payloads,f):
    """Bounded full-byte alias/base equality; neither F32 tolerance nor padding skip."""
    for alias,base in pairs:
        ap=f.safe_file(root,alias['file']);bp=f.safe_file(root,base['file']);before=identity(ap)
        if str(bp)not in payloads or identity(bp)!=payloads[str(bp)]or before[2]!=alias['bytes']or bp.stat().st_size!=base['bytes']:raise ValueError('actual alias/base payload size or canonical identity differs')
        digest=hashlib.sha256();remaining=alias['bytes']
        with ap.open('rb')as a,bp.open('rb')as b:
            while remaining:
                count=min(1<<20,remaining);x=a.read(count);y=b.read(count)
                if len(x)!=count or len(y)!=count or x!=y:raise ValueError('known KDA alias raw bytes differ from canonical tensor')
                digest.update(x);remaining-=count
            if a.read(1)or b.read(1):raise ValueError('alias/base raw payload exceeds declared size')
        if digest.hexdigest()!=alias['sha256']or digest.hexdigest()!=base['sha256']or identity(ap)!=before or identity(bp)!=payloads[str(bp)]:raise ValueError('known KDA alias raw SHA/identity differs')
        payloads[str(ap)]=before
    return {'distinct_alias_names':len({a['name']for a,b in pairs}),'verified_alias_rows':len(pairs),
        'verified_batches':len({(a['offset'],a['input_tokens'],a['batch_index'])for a,b in pairs}),
        'aliases_per_batch':34,'canonical_series_unchanged':118,'metadata_and_full_raw_bytes_exact':True,
        'scope':'only exact pinned34 KDA reshape aliases; not additional numerical series or model qualification'}

def window_rows(root,report,m,*,gpu):
    f=m['fidelity'];names=m['native'].EXPECTED_NAMES
    if len(names)!=118:raise ValueError('fixed118-series GLM contract changed')
    phase='probe128' if gpu else 'full';metadata=f.safe_file(root,'tensors.jsonl');batchfile=f.safe_file(root,'batches.jsonl')
    if metadata.stat().st_size>64<<20 or batchfile.stat().st_size>64<<20:raise ValueError('bounded closed metadata required')
    for path in (metadata,batchfile):
        with path.open('rb') as stream:
            if path.stat().st_size:stream.seek(-1,2)
            if stream.read()!=b'\n':raise ValueError('closed complete JSONL required; no live/truncated rows')
    all_batches=[r for r in f.json_lines(batchfile) if 'offset'in r]
    if gpu and any(r.get('request')!=0 or r.get('phase')!=phase or type(r.get('offset')) is not int or not 0<=r['offset']<128 for r in all_batches):raise ValueError('candidate computed outside request0/window128')
    batches=[r for r in all_batches if r.get('request')==0 and r.get('phase')==phase and r['offset']<128]
    offset=0;wanted=[]
    for index,row in enumerate(batches):
        if type(row.get('tokens')) is not int or row.get('offset')!=offset or row.get('returncode')!=0 or row['tokens']<=0 or offset+row['tokens']>128:raise ValueError('incomplete/overlapping/nonzero window batches')
        take=min(report['microbatch'],128-offset)
        if row['tokens']!=take:raise ValueError('physical128-window batch geometry differs')
        if gpu and (row.get('decode_end')!=128 or row.get('prefill_chunk_policy')!=m['gpu'].TAIL_POLICY or row.get('aligned_take')!=take or row.get('chunk16_split') is not False or row.get('odd_tail_split') is not False):raise ValueError('actual v4/probe batch geometry differs')
        wanted.append((offset,take,index));offset+=take
    if offset!=128:raise ValueError('window batch coverage must be exactly0..127')
    result={name:[] for name in names};payloads={};alias_pairs=[]
    rows=f.json_lines(metadata)
    if gpu:rows,alias_pairs=candidate_alias_catalog(rows,wanted,names)
    for row in rows:
        if gpu and (row.get('request')!=0 or row.get('phase')!=phase or row.get('name') not in names or type(row.get('offset')) is not int or not 0<=row['offset']<128):raise ValueError('candidate tensor is outside the declared118-series probe')
        if row.get('request')!=0 or row.get('phase')!=phase or row.get('name') not in names or row.get('offset',128)>=128:continue
        name=row['name'];label=f.family(name);axis=m['batch'].TOKEN_AXES[label]
        interval=(row.get('offset'),row.get('input_tokens'),row.get('batch_index'))
        if interval not in wanted or row.get('type')!=(26 if label=='routes' else 0):raise ValueError('tensor dtype/batch metadata differs')
        shape=[8,row['input_tokens'],1,1] if label=='routes' else [288,row['input_tokens'],1,1] if label=='router' else [128,64,row['input_tokens'],1]
        if row.get('shape')!=shape:raise ValueError('actual pinned GLM tensor shape differs: '+name)
        path=f.safe_file(root,row['file']);before=identity(path)
        if type(row.get('bytes')) is not int or not 0<row['bytes']<=64<<20 or sha(path)!=row.get('sha256'):raise ValueError('raw tensor SHA/size invalid')
        view=f.tensor_view(root,row)
        if view.shape[axis]!=row['input_tokens']:raise ValueError('strided logical token axis differs')
        del view
        if identity(path)!=before:raise ValueError('raw tensor changed during validation')
        payloads[str(path)]=before;result[name].append(row)
    for name,rows in result.items():
        rows.sort(key=lambda r:r['offset'])
        if [(r['offset'],r['input_tokens'],r['batch_index']) for r in rows]!=wanted:raise ValueError('missing/duplicate118-series coverage: '+name)
    aliases=verify_alias_payloads(root,alias_pairs,payloads,f)if gpu else None
    return result,payloads,aliases

def metal_window(reference,candidate,a_rows,b_rows,name,m):
    """Ancillary original Metal tolerance; never substitutes for CPU bit parity."""
    f=m['fidelity'];axis=m['batch'].TOKEN_AXES[f.family(name)]
    ai=bi=ac=bc=0
    out={'passed':True,'tokens':0,'elements':0,'tolerance_mismatches':0,'bit_mismatches':0,
         'nonfinite_reference':0,'nonfinite_candidate':0,'max_abs_error':0.0}
    while ai<len(a_rows) and bi<len(b_rows):
        ar,br=a_rows[ai],b_rows[bi]
        if ar['offset']+ac!=br['offset']+bc:raise ValueError('Metal diagnostic logical position differs')
        count=min(ar['input_tokens']-ac,br['input_tokens']-bc)
        av,bv=f.tensor_view(reference,ar),f.tensor_view(candidate,br)
        sa,sb=[slice(None)]*4,[slice(None)]*4;sa[axis]=slice(ac,ac+count);sb[axis]=slice(bc,bc+count)
        check=f.compare_arrays(av[tuple(sa)],bv[tuple(sb)],f.CRITERIA['metal'],integer=f.family(name)=='routes')
        out['passed']&=check['passed'];out['tokens']+=count;out['elements']+=check['elements']
        out['tolerance_mismatches']+=check['tolerance_mismatch_count'];out['bit_mismatches']+=check['bit_mismatch_count']
        out['nonfinite_reference']+=check['nonfinite_reference'];out['nonfinite_candidate']+=check['nonfinite_candidate']
        maximum=check['max_abs_error_finite_pairs']
        if maximum is not None:out['max_abs_error']=max(out['max_abs_error'],maximum)
        del av,bv
        ac+=count;bc+=count
        if ac==ar['input_tokens']:ai+=1;ac=0
        if bc==br['input_tokens']:bi+=1;bc=0
    if ai!=len(a_rows) or bi!=len(b_rows) or out['tokens']!=128:raise ValueError('Metal diagnostic incomplete token coverage')
    return out

def checkpoint128(root,m,*,gpu=False):
    path=m['fidelity'].safe_file(root,'state_checkpoints.jsonl')
    if path.stat().st_size>64<<20:raise ValueError('bounded closed checkpoint metadata required')
    with path.open('rb') as stream:
        if path.stat().st_size:stream.seek(-1,2)
        if stream.read()!=b'\n':raise ValueError('closed checkpoint JSONL required')
    all_rows=m['fidelity'].json_lines(path)
    if gpu:
        # No candidate checkpoint is silently discarded as outside the probe.
        if len(all_rows)!=1:raise ValueError('candidate checkpoint coverage must be exactly {(0,128)}')
        rows=all_rows
    else:
        # The historical CPU control contains complete requests/checkpoints.
        rows=[r for r in all_rows if r.get('request')==0 and r.get('token_end')==128]
    if len(rows)!=1:raise ValueError('one explicit checkpoint128 required')
    row=rows[0]
    if (type(row.get('request')) is not int or row['request']!=0 or
            type(row.get('token_end')) is not int or row['token_end']!=128 or
            type(row.get('next_position')) is not int or row['next_position']!=128 or
            row.get('status')!='hashed' or type(row.get('state_seq_flags')) is not int or row['state_seq_flags']!=0 or
            type(row.get('bytes')) is not int or not 0<row['bytes']<=256<<20 or
            type(row.get('sha256')) is not str or re.fullmatch('[0-9a-f]{64}',row['sha256']) is None):raise ValueError('invalid explicit checkpoint128')
    return {'bytes':row['bytes'],'sha256':row['sha256'],'next_position':128,'scope':'native opaque checkpoint hash only; no final prompt or raw checkpoint export'}

def compare(reference,candidate,reference_guard,candidate_guard,binding_path):
    result={'schema_version':1,'status':'invalid_probe','window_identical':False,'early_rejection':True,'tokens':128,'complete':False,
        'full_qualification':False,'full_model_qualified':False,'snapshot_qualified':False,'performance_qualified':False,'EOS_qualified':False,
        'scope':'closed reject-only128 window; two actual builds/own keys/clear guards/118 raw series; never a full arm','checks':[],'errors':[]}
    try:
        prepared=read(PREPARE);m=modules(prepared);binding,ab,bb,profile=load_binding(binding_path,prepared,m)
        original=read(checked_file(binding['frozen_prompts']));agents=m['gpu'].validate_full_prompts(original,sha(FROZEN))
        roots=[Path(reference).resolve(),Path(candidate).resolve()]
        if roots[0]==roots[1]:raise ValueError('distinct actual CPU/GPU capture roots required')
        reports=[read(root/'report.json') for root in roots];guards=[pinned_reference_guard(reference_guard,prepared),read(candidate_guard)]
        tracks=[Path(reference_guard),Path(candidate_guard),roots[0]/'report.json',roots[1]/'report.json',Path(binding_path),PREPARE]
        for root,guard in zip(roots,(reference_guard,candidate_guard)):
            tracks.extend([root/'tensors.jsonl',root/'batches.jsonl',root/'state_checkpoints.jsonl',Path(guard).parent/'run-prompts.json'])
        tracks.append(Path(candidate_guard).parent/'native-stderr.log')
        tracked={str(path):identity(path) for path in tracks}
        normalized=[];keys=[];model_paths=[]
        for i in (0,1):
            closed_guard(guards[i],gpu=bool(i),m=m)
            norm,key,model=dependencies(roots[i],(reference_guard,candidate_guard)[i],reports[i],guards[i],binding[('reference','candidate')[i]],(ab,bb)[i],m,original,profile if i else None)
            normalized.append(norm);keys.append(key);model_paths.append(model)
            ids=agents[1]['token_ids'];expected_hash=hashlib.sha256(struct.pack('<'+'i'*len(ids),*ids)).hexdigest()
            if reports[i]['requests'][0].get('prompt_tokens')!=len(ids) or reports[i]['requests'][0].get('prompt_ids_sha256')!=expected_hash:raise ValueError('full request0 token identity differs')
        if normalized[0]!=normalized[1] or model_paths[0]!=model_paths[1]:raise ValueError('model/tokenizer/template/other math/environment dependencies differ beyond explicitly bound changes')
        if guards[1].get('gpu_execution_profile')!=profile or guards[1].get('actual_gpu') is not True:raise ValueError('actual GPU profile/execution required')
        stderr=Path(candidate_guard).parent/'native-stderr.log'
        if stderr.stat().st_size>32<<20:raise ValueError('bounded raw GPU trace required')
        trace=m['gpu'].parse_trace(stderr.read_text(errors='replace'),profile,baseline_swap=guards[1]['baseline']['swap_used_bytes'])
        if trace!=guards[1].get('gpu_trace') or guards[1].get('actual_backend')!=trace['actual_backend'] or reports[1]['microbatch'] not in trace['multi_token_shapes']:raise ValueError('raw multi-token/terminal/alias trace differs')
        if list(roots[1].glob('logits-*.f32')) or list(roots[1].glob('continuation-*.f32')):raise ValueError('candidate unexpectedly exported logits/continuation')
        for name in ('continuation_tensors.jsonl','continuation_states.jsonl'):
            path=m['fidelity'].safe_file(roots[1],name)
            if path.stat().st_size:raise ValueError('candidate continuation capture must remain empty')
        ar,ap,_=window_rows(roots[0],reports[0],m,gpu=False);br,bp,aliases=window_rows(roots[1],reports[1],m,gpu=True)
        result['candidate_KDA_aliases']=aliases
        checks=[];metal_checks=[]
        for name in sorted(m['native'].EXPECTED_NAMES):
            verify_sources(prepared)
            check=m['batch']._token_compare(roots[0],roots[1],ar[name],br[name],name)
            if check['tokens']!=128:raise ValueError('stitched logical coverage differs')
            checks.append({'name':name,**check})
            metal_checks.append({'name':name,**metal_window(roots[0],roots[1],ar[name],br[name],name,m)})
        checkpoints=[checkpoint128(roots[0],m),checkpoint128(roots[1],m,gpu=True)]
        checkpoint_equal=checkpoints[0]==checkpoints[1]
        for path,value in {**tracked,**ap,**bp}.items():
            if identity(path)!=value:raise ValueError('capture/provenance changed during comparison')
        verify_sources(prepared)
        identical=all(c['passed'] for c in checks) and checkpoint_equal
        result.update(status='identical_probe_window' if identical else 'rejected_probe_window',window_identical=identical,early_rejection=not identical,
            complete_window=True,expected_tensor_names=118,checks=checks,checkpoint128_exact=checkpoint_equal,checkpoints=checkpoints,
            two_actual_builds_verified=True,two_own_keys_recomputed=keys,two_keys_expected_to_differ=keys[0]!=keys[1],
            guards_revalidated=True,gpu_trace=trace,gpu_trace_sha256=sha(stderr),binding_sha256=sha(binding_path),
            cpu_bit_contract={'bitexact_f32':True,'atol':0.0,'rtol':0.0},
            metal_tolerance_checks=metal_checks,metal_window_allclose=all(c['passed'] for c in metal_checks),
            metal_tolerance_contract={'atol':1e-3,'rtol':1e-4,'scope':'ancillary original Metal tolerance, routes exact; never substituted for the strict128 bit/checkpoint rejection test'},
            opaque_checkpoint_scope='explicit128 only; no complete prompt/finalstate/EOS claim')
    except (OSError,ValueError,KeyError,TypeError,ImportError) as error:result['errors'].append(str(error))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--make-binding',action='store_true')
    parser.add_argument('--driver-build',type=Path,default=BUILD_SUPERVISOR/'build-run1/driver-build.json')
    parser.add_argument('--provider-build',type=Path,default=WORK/'metal-tensorops-q8-nocopy/build-run1/provider-build.json')
    parser.add_argument('--profile',type=Path,default=BUILD_SUPERVISOR/'build-run1/gpu-profile.json')
    for name in ('reference','candidate','reference-guard','candidate-guard','binding'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.make_binding:
        prepared=read(PREPARE);value=make_binding(prepared,modules(prepared),args.driver_build,args.provider_build,args.profile)
    else:
        if any(getattr(args,name) is None for name in ('reference','candidate','reference_guard','candidate_guard','binding')):raise ValueError('both closed captures, guards and actual binding required')
        value=compare(args.reference,args.candidate,args.reference_guard,args.candidate_guard,args.binding)
    args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return 0 if args.make_binding or value['status']=='identical_probe_window' else 2

if __name__=='__main__':raise SystemExit(main())
