"""Strict two-build comparison of the isolated CPU8-compatible-tail candidate.

Existing comparators and measured receipts are not modified. A distinct binding
is emitted only after an actual build. Raw complete captures are checked with
the existing guards, CPU bit-exact criteria and continuation/replay functions.
"""
from __future__ import annotations
import argparse
import copy
import difflib
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent/'airllm-next'
sys.path[:0]=[str(REPO),str(REPO/'air_llm')]

POLICY='cpu8-tail-compatible-v1'
REFERENCE_BINARY='62a7d2578c58e96839be2d4c79d8c414c8d535e5a1adac54820043f9e695f4b0'
REFERENCE_SOURCE='558aa441ed74aca6b295613ccb414c4b1f2bf2134a6bdc0ae32a09b35c32c395'
CANDIDATE_SOURCE='dac2113041898470e2ea4eaf43af51da1132d798430117e0b1ff67862c926433'
POLICY_PARAMETERS={'prefill_chunk_policy':POLICY,'prefill_chunk_policy_version':1,'tail_reference_microbatch':8}
DELEGATED_MODULE_PATHS={
    'research.glm53_batched_prefill.native_compare':'research/glm53_batched_prefill/native_compare.py',
    'research.glm53_batched_prefill.batch_compare':'research/glm53_batched_prefill/batch_compare.py',
    'research.glm53_m5.compare_fidelity':'research/glm53_m5/compare_fidelity.py',
    'research.glm53_batched_prefill.snapshot_key':'research/glm53_batched_prefill/snapshot_key.py',
    'air_llm.airllm.nextgen.glm_native':'air_llm/airllm/nextgen/glm_native.py',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def verify_module_files(repository,pins,loaded_modules=None):
    if set(pins)!=set(DELEGATED_MODULE_PATHS):raise ValueError('complete delegated strict-module pin set required')
    checked={}
    cache=sys.modules if loaded_modules is None else loaded_modules
    for name,relative in DELEGATED_MODULE_PATHS.items():
        path=(Path(repository)/relative).resolve()
        if sha(path)!=pins[name]:raise ValueError('delegated strict module source changed: '+name)
        cached=cache.get(name)
        if cached is not None and Path(getattr(cached,'__file__','')).resolve()!=path:
            raise ValueError('delegated strict module was loaded from another source: '+name)
        checked[name]=pins[name]
    return checked

def verify_delegated_modules():
    prepared=read(ROOT/'prepare-manifest.json')
    return verify_module_files(REPO,prepared['delegated_module_hashes'])

def canonical_key(manifest):
    verify_delegated_modules()
    module=importlib.import_module('research.glm53_batched_prefill.snapshot_key')
    verify_delegated_modules()
    value=module.canonical_key(manifest)
    verify_delegated_modules()
    return value

def checked_delegate(function):
    def checked(*args,**kwargs):
        verify_delegated_modules()
        value=function(*args,**kwargs)
        verify_delegated_modules()
        return value
    return checked

def _hash(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('actual lowercase SHA256 identity required')
def _input(build,suffix):
    values=[(Path(path),digest) for path,digest in build['inputs_sha256'].items() if path.endswith('/'+suffix)]
    if len(values)!=1:raise ValueError('exact unique build input required: '+suffix)
    path,digest=values[0]
    if sha(path)!=digest:raise ValueError('actual build source/input bytes differ: '+suffix)
    return path,digest

def load_build_pair(reference_path,candidate_path,binding_path):
    verify_delegated_modules()
    prepared=read(ROOT/'prepare-manifest.json');binding=read(binding_path)
    if type(binding.get('schema_version')) is not int or binding.get('schema_version')!=1 or binding.get('policy')!=POLICY or binding.get('status')!='built_bound_not_numerically_qualified':
        raise ValueError('candidate must be actually built and explicitly bound before comparison')
    expected={'reference':(REFERENCE_BINARY,REFERENCE_SOURCE),'candidate':(binding.get('candidate',{}).get('binary_sha256'),CANDIDATE_SOURCE)}
    for side,(binary,source) in expected.items():
        _hash(binary);_hash(source)
        if binding.get(side,{}).get('binary_sha256')!=binary or binding[side].get('helper_source_sha256')!=source:
            raise ValueError('two-build helper identity binding differs: '+side)
    if binding['reference'].get('build_receipt_sha256')!=prepared['original_build_receipt_sha256']:
        raise ValueError('original measured build receipt identity differs')
    paths={'reference':Path(reference_path),'candidate':Path(candidate_path)}
    result={'binding':binding,'builds':{}}
    for side,path in paths.items():
        if sha(path)!=binding[side].get('build_receipt_sha256'):
            raise ValueError('build manifest bytes do not match explicit binding: '+side)
        build=read(path)
        if type(build.get('returncode')) is not int or build['returncode']!=0:
            raise ValueError('successful real build returncode required: '+side)
        if side=='candidate' and (build.get('status')!='built_metadata_only_smoked' or build.get('build_performed') is not True or
                build.get('binary_rebuilt') is not True or build.get('runtime_hashes_verified_for_this_build') is not True or
                any(build.get(name) is not True for name in ('tracked_inputs_verified_before_compile','tracked_inputs_verified_after_compile','tracked_inputs_verified_after_smoke')) or
                build.get('model_loaded') is not False or type(build.get('gpu_commands')) is not int or build.get('gpu_commands')!=0 or build.get('policy')!=POLICY or
                build.get('original_build_receipt_sha256')!=binding['reference']['build_receipt_sha256']):
            raise ValueError('candidate build/metadata provenance is incomplete or false')
        if build.get('binary_sha256')!=binding[side]['binary_sha256']:
            raise ValueError('build binary differs from the separately bound actual hash: '+side)
        command=build.get('command')
        if not isinstance(command,list) or command.count('-o')!=1:
            raise ValueError('exact helper compile output is required')
        binary_path=Path(command[command.index('-o')+1])
        if not binary_path.is_file() or sha(binary_path)!=build['binary_sha256']:
            raise ValueError('actual helper binary bytes differ from build: '+side)
        source_path,source_digest=_input(build,'glm53_batched_prefill/native_prefill.cpp')
        if source_digest!=binding[side]['helper_source_sha256']:
            raise ValueError('helper source differs from the exact variant: '+side)
        for suffix,key in [('glm53_batched_prefill/snapshot.h','snapshot_source'),('glm53_gpu_v3/state_capture.inc','state_capture_source')]:
            _,digest=_input(build,suffix)
            if digest!=binding['shared_source_sha256'][key]:raise ValueError('unchanged dependency differs: '+key)
        if build.get('native_base_revision')!=prepared['native_base_revision'] or build.get('native_libraries_sha256')!=binding['shared_native_libraries_sha256']:
            raise ValueError('native revision or actual library hashes differ')
        if side=='candidate':
            _,builder_sha=_input(build,'build_helper.py')
            if builder_sha!=sha(ROOT/'build_helper.py'):raise ValueError('candidate build driver identity differs')
            reconstruction=read(prepared['native_source_reconstruction_receipt'])
            if build.get('source_reconstruction_receipt_sha256')!=prepared['native_source_reconstruction_receipt_sha256'] or build.get('native_sources_exact')!={name:item['expected_sha256'] for name,item in reconstruction['source_checks'].items()}:
                raise ValueError('candidate exact native reconstruction differs')
            header_names=('include/llama.h','ggml/include/ggml.h','ggml/include/ggml-backend.h','vendor/nlohmann/json.hpp')
            if set(build.get('header_paths',{}))!=set(header_names) or set(build.get('header_sha256',{}))!=set(header_names):
                raise ValueError('complete tracked helper header provenance required')
            for relative in header_names:
                actual_path,actual_digest=_input(build,relative)
                if str(actual_path)!=build['header_paths'][relative] or actual_digest!=build['header_sha256'][relative]:
                    raise ValueError('candidate actual tracked header differs from declaration: '+relative)
                if relative=='vendor/nlohmann/json.hpp':
                    original=prepared['reference_vendor_json_header']
                    if sha(original['path'])!=original['sha256'] or actual_digest!=original['sha256']:
                        raise ValueError('candidate JSON header differs from pinned baseline source')
                else:
                    expected_header=[digest for name,digest in result['builds']['reference']['inputs_sha256'].items() if name.endswith('/'+relative)]
                    if len(expected_header)!=1 or actual_digest!=expected_header[0]:
                        raise ValueError('candidate helper ABI header differs: '+relative)
        result['builds'][side]=build
        result.setdefault('source_paths',{})[side]=source_path
    if result['builds']['reference']['native_libraries_sha256']!=prepared['runtime_libraries_sha256'] or binding['shared_native_libraries_sha256']!=prepared['runtime_libraries_sha256']:
        raise ValueError('native library hashes differ from the measured runtime')
    if binding['shared_source_sha256']!={k:v for k,v in prepared['reference_source_identity'].items() if k!='helper_source'}:
        raise ValueError('wrapper/runtime/snapshot/state_capture source identities changed')
    patch=ROOT/'patches/native-prefill-tail-even16.patch'
    if sha(patch)!=binding.get('patch_sha256') or sha(patch)!=prepared['patch_sha256']:
        raise ValueError('variant patch identity differs')
    actual=''.join(difflib.unified_diff(result['source_paths']['reference'].read_text().splitlines(True),result['source_paths']['candidate'].read_text().splitlines(True),
        fromfile='a/research/glm53_batched_prefill/native_prefill.cpp',tofile='b/research/glm53_batched_prefill/native_prefill.cpp'))
    if actual!=patch.read_text():raise ValueError('actual two sources do not match the exact scheduling-only patch')
    return result

def arm_dependencies(report,receipt,build,pair,side):
    manifest=receipt['preflight']['snapshot_manifest'];binding=pair['binding']
    key,_=canonical_key(manifest)
    if key!=receipt['preflight'].get('snapshot_key_sha256') or key!=report.get('snapshot_key_sha256'):
        raise ValueError('own snapshot key was not recomputed from its actual dependency manifest: '+side)
    sources=receipt['preflight'].get('source_sha256')
    expected_sources={**binding['shared_source_sha256'],'helper_source':binding[side]['helper_source_sha256']}
    if sources!=expected_sources:
        raise ValueError('captured helper/runtime/wrapper/snapshot/state_capture source hashes differ: '+side)
    combined=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if manifest['runtime_source_sha256']!=combined:
        raise ValueError('runtime source compound hash differs: '+side)
    binaries={'helper':binding[side]['binary_sha256'],**binding['shared_native_libraries_sha256']}
    if manifest['runtime_binaries']!=binaries or receipt['preflight'].get('runtime_binaries')!=binaries:
        raise ValueError('captured binary/library hashes differ from exact two builds: '+side)
    if report.get('state_parameters')!=manifest['state_parameters']:
        raise ValueError('captured native state parameters differ from keyed parameters: '+side)
    params=manifest['state_parameters'];ub=report.get('microbatch')
    if ub not in (8,16,32,64,128) or report.get('logical_batch')!=params.get('n_batch') or ub!=params.get('n_ubatch'):
        raise ValueError('physical/logical microbatch metadata differs: '+side)
    if side=='reference' and ub!=8:raise ValueError('the admitted reference profile is the complete CPU8 control')
    if side=='candidate':
        smoke=build.get('parameter_smoke',{}).get(str(ub),{})
        if type(smoke.get('returncode')) is not int or smoke.get('returncode')!=0 or json.loads(smoke.get('stdout','null'))!=params or smoke.get('state_parameters')!=params:
            raise ValueError('candidate state parameters do not match its actual binary metadata smoke')
        if any(type(params.get(k)) is not type(v) or params.get(k)!=v for k,v in POLICY_PARAMETERS.items()) or report.get('prefill_chunk_policy')!=POLICY or report.get('tail_reference_microbatch')!=8 or report.get('prefill_chunk_policy_qualified') is not False:
            raise ValueError('candidate tail policy is absent or different')
    else:
        if build.get('parameter_smoke',{}).get(str(ub))!=params or any(name in params for name in POLICY_PARAMETERS):
            raise ValueError('reference is not the original measured helper parameter set')
        if report.get('prefill_chunk_policy') is not None:raise ValueError('reference unexpectedly declares a tail variant')
    normalized=copy.deepcopy(manifest)
    for name in ('n_batch','n_ubatch'):del normalized['state_parameters'][name]
    if side=='candidate':
        for name in POLICY_PARAMETERS:del normalized['state_parameters'][name]
    # Normalize only identities already independently checked against the two
    # build receipts, actual files and exact patch. Nothing is ignored blindly.
    reference_sources={**binding['shared_source_sha256'],'helper_source':binding['reference']['helper_source_sha256']}
    normalized['runtime_source_sha256']=hashlib.sha256(json.dumps(reference_sources,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    normalized['runtime_binaries']['helper']=binding['reference']['binary_sha256']
    return normalized


def check_policy_chunks(report,batches,side):
    ub=report['microbatch']
    for request,row in enumerate(report['requests']):
        expected=[];offset=0;end=row['prompt_tokens']
        while offset<end:
            raw=min(ub,end-offset)
            take=raw-raw%8 if side=='candidate' and raw>8 and raw%2 else raw
            expected.append((offset,take,raw));offset+=take
        selected=[item for item in batches if item['request']==request]
        if [(item['offset'],item['tokens']) for item in selected]!=[(offset,take) for offset,take,_ in expected]:
            raise ValueError('actual batch receipts violate the declared source policy: '+side)
        if side=='candidate':
            for item,(_,take,raw) in zip(selected,expected):
                if type(item.get('unadjusted_take')) is not int or item.get('unadjusted_take')!=raw or item.get('odd_tail_split') is not (take!=raw) or item.get('prefill_chunk_policy')!=POLICY:
                    raise ValueError('candidate batch policy receipt is missing or false')


def compare(reference_root, candidate_root, reference_guard, candidate_guard, reference_build, candidate_build, binding):
    reference_root, candidate_root = Path(reference_root), Path(candidate_root)
    result = {"schema_version": 1, "passed": False, "full_model_qualified": False, "performance_qualified": False, "errors": [], "criteria": None,
              "scope": "Exact two-build CPU8-tail-policy experiment; CPU10/GPU0 full prefill across batch sizes; per-token ordered routes/router/KDA, finite full final logits, raw final sequence states and complete-state hashes at common boundaries; no EOS or private-history qualification",
              "cross_batch_write_event_parity": "not asserted: physical chunks differ; require per-batch target coverage and complete states at common token boundaries",
              "logits": [], "final_states": [], "checkpoints": [], "tensors": [], "coverage": {}}
    try:
        verify_delegated_modules()
        from research.glm53_m5.compare_fidelity import CRITERIA, compare_arrays, family
        from research.glm53_batched_prefill.native_compare import _read, _check_replay, _compare_writes, _compare_continuation
        from research.glm53_batched_prefill.batch_compare import _checkpoints, _checkpoint_replay, _series, _token_compare, TOKEN_AXES
        import numpy as np
        verify_delegated_modules()
        _read=checked_delegate(_read);_check_replay=checked_delegate(_check_replay)
        _compare_writes=checked_delegate(_compare_writes);_compare_continuation=checked_delegate(_compare_continuation)
        _checkpoints=checked_delegate(_checkpoints);_checkpoint_replay=checked_delegate(_checkpoint_replay)
        _series=checked_delegate(_series);_token_compare=checked_delegate(_token_compare)
        compare_arrays=checked_delegate(compare_arrays);family=checked_delegate(family)
        result['criteria']=CRITERIA['cpu'];result['coverage']={k:0 for k in TOKEN_AXES}
        pair=load_build_pair(reference_build,candidate_build,binding)
        a, at, ab, ag, aw, a_cont = _read(reference_root, reference_guard)
        b, bt, bb, bg, bw, b_cont = _read(candidate_root, candidate_guard)
        if a["mode"] != "full" or b["mode"] != "full":
            raise ValueError("cross-batch comparison requires full/full uncached prefill")
        if arm_dependencies(a,ag,pair['builds']['reference'],pair,'reference') != arm_dependencies(b,bg,pair['builds']['candidate'],pair,'candidate'):
            raise ValueError("dependencies differ beyond the exact bound helper policy and batch sizes")
        check_policy_chunks(a,ab,'reference');check_policy_chunks(b,bb,'candidate')
        result['build_binding']={'reference_binary_sha256':pair['binding']['reference']['binary_sha256'],
            'candidate_binary_sha256':pair['binding']['candidate']['binary_sha256'],'policy':POLICY,'validated':True}
        result["continuation"] = _compare_continuation(reference_root, candidate_root, a, b, a_cont, b_cont)
        result["continuation_replays"] = {
            "reference": _compare_continuation(reference_root, reference_root, a, a, a_cont, a_cont, ((0, 2),)),
            "candidate": _compare_continuation(candidate_root, candidate_root, b, b, b_cont, b_cont, ((0, 2),)),
        }
        for field in ("vocab_size", "threads", "gpu_layers", "physical_slots"):
            if a[field] != b[field]:
                raise ValueError("incompatible native profile: " + field)
        ac, bc = _checkpoints(reference_root, a), _checkpoints(candidate_root, b)
        result["same_agent_replays"] = {
            "reference": _check_replay(reference_root, a, at, ab, aw),
            "candidate": _check_replay(candidate_root, b, bt, bb, bw),
        }
        result["checkpoint_replays"] = {"reference": _checkpoint_replay(ac), "candidate": _checkpoint_replay(bc)}
        # Self-comparison validates complete target coverage, not cross-batch parity.
        result["observed_write_coverage"] = {
            "reference": _compare_writes(aw, aw, [(r["request"], r["offset"]) for r in ab], False),
            "candidate": _compare_writes(bw, bw, [(r["request"], r["offset"]) for r in bb], False),
        }
        common_checkpoints = set(ac) & set(bc)
        for key in sorted(common_checkpoints):
            result["checkpoints"].append({"request": key[0], "token_end": key[1],
                "passed": ac[key]["bytes"] == bc[key]["bytes"] and ac[key]["sha256"] == bc[key]["sha256"]})
        result["checkpoint208_compared"] = any(end == 208 for _, end in common_checkpoints)
        for request, (ar, br) in enumerate(zip(a["requests"], b["requests"])):
            for field in ("prompt_tokens", "prompt_ids_sha256", "next_position"):
                if ar[field] != br[field]:
                    raise ValueError("full native prompts or positions differ")
            result["logits"].append({"request": request, **compare_arrays(
                np.memmap(reference_root / f"logits-{request}.f32", dtype="<f4", mode="r"),
                np.memmap(candidate_root / f"logits-{request}.f32", dtype="<f4", mode="r"), CRITERIA["cpu"])})
            result["final_states"].append({"request": request,
                "passed": ar["sequence_bytes"] == br["sequence_bytes"] and ar["sequence_sha256"] == br["sequence_sha256"]})
        a_series, b_series = _series(a, at, ab), _series(b, bt, bb)
        for request, name in sorted(a_series):
            check = _token_compare(reference_root, candidate_root, a_series[request, name], b_series[request, name], name)
            result["tensors"].append({"request": request, "name": name, **check})
            result["coverage"][family(name)] += check["elements"]
        result["passed"] = bool(result["continuation"]["passed"] and all(r["passed"] for r in result["continuation_replays"].values()) and
            all(r["passed"] for r in result["same_agent_replays"].values()) and
            all(result["checkpoint_replays"].values()) and all(r["passed"] for r in result["observed_write_coverage"].values()) and
            all(r["passed"] for r in result["logits"] + result["final_states"] + result["checkpoints"] + result["tensors"]))
        result["reference_microbatch"], result["candidate_microbatch"] = a["microbatch"], b["microbatch"]
        result['delegated_module_sha256']=verify_delegated_modules()
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        result["passed"]=False
        result["errors"].append(str(error))
    return result



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path);parser.add_argument('candidate',type=Path)
    for name in ('reference-guard','candidate-guard','reference-build','candidate-build','binding','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.reference,args.candidate,args.reference_guard,args.candidate_guard,
        args.reference_build,args.candidate_build,args.binding)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
