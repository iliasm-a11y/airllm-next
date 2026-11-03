"""Early rejection only; an incomplete arm can never pass full qualification."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

class IncompleteWindow(ValueError):
    pass


def _load_dependencies(root, tokens, canonical_key):
    # Preflight is already written before native load. This does not validate
    # live guard outcomes; it merely avoids comparing different models/prompts.
    guard=json.loads((root.parent/'guard.json').read_text())
    manifest=guard['preflight']['snapshot_manifest']
    key,_=canonical_key(manifest)
    if key!=guard['preflight'].get('snapshot_key_sha256'):
        raise ValueError('preflight dependency key does not match its manifest')
    params=manifest['state_parameters']
    if params.get('n_gpu_layers')!=0 or params.get('n_threads')!=10 or params.get('n_threads_batch')!=10:
        raise ValueError('early comparison requires CPU10/GPU0 state parameters')
    if len(manifest['prefix_tokens'])<tokens:
        raise ValueError('window exceeds the identified common native prefix')
    normalized=json.loads(json.dumps(manifest,allow_nan=False))
    for field in ('n_batch','n_ubatch'):
        del normalized['state_parameters'][field]
    return normalized


def compare(args):
    sys.path.insert(0,str(args.repo.resolve()))
    from research.glm53_batched_prefill.batch_compare import _token_compare, TOKEN_AXES
    from research.glm53_batched_prefill.native_compare import EXPECTED_NAMES
    from research.glm53_batched_prefill.snapshot_key import canonical_key
    from research.glm53_m5.compare_fidelity import CRITERIA, family, safe_file, tensor_view
    result={'schema_version':1,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'early rejection diagnostic on request0 only; no complete arm, checkpoint, guard, EOS or speed qualification',
        'full_qualification':False,'criteria':CRITERIA['cpu'],'tokens':args.tokens,
        'expected_tensor_names':len(EXPECTED_NAMES),'complete_window':False,
        'window_identical':False,'early_rejection':False,'status':'invalid_capture','checks':[],'errors':[]}
    try:
        if type(args.tokens) is not int or args.tokens not in (16,32,64,128):
            raise ValueError('bounded complete token window required')
        if len(EXPECTED_NAMES)!=118:
            raise ValueError('the fixed118-name GLM coverage contract changed')
        if _load_dependencies(args.reference,args.tokens,canonical_key)!=_load_dependencies(args.candidate,args.tokens,canonical_key):
            raise ValueError('model/prompt/state dependencies differ beyond batch sizes')
        def selected(root):
            path=root/'tensors.jsonl'
            if path.stat().st_size>64*1024**2:
                raise ValueError('early tensor manifest exceeds64MiB')
            raw=path.read_bytes()
            # A currently-written trailing JSON line is not usable evidence.
            lines=raw[:raw.rfind(b'\n')+1].splitlines()
            rows_by_name={name:[] for name in EXPECTED_NAMES}
            for line in lines:
                if len(line)>2_000_000:
                    raise ValueError('oversize tensor metadata row')
                row=json.loads(line)
                if row.get('request')!=0 or row.get('phase')!='full' or row.get('name') not in rows_by_name:
                    continue
                offset,count=row.get('offset'),row.get('input_tokens')
                if type(offset) is not int or offset<0 or type(count) is not int or count<=0:
                    raise ValueError('invalid logical token interval')
                if offset+count>args.tokens:
                    continue
                name=row['name'];label=family(name);axis=TOKEN_AXES[label]
                if row.get('type')!=(26 if label=='routes' else 0):
                    raise ValueError('unexpected original tensor dtype: '+name)
                view=tensor_view(root,row) # Checks actual GGML byte strides/span.
                if view.shape[axis]!=count or view.shape[3]!=1:
                    raise ValueError('declared input token count differs from actual strided token axis: '+name)
                del view
                if not isinstance(row.get('sha256'),str) or re.fullmatch('[0-9a-f]{64}',row['sha256']) is None:
                    raise ValueError('missing payload SHA256: '+name)
                payload=safe_file(root,row['file'])
                before=payload.stat()
                digest=hashlib.sha256()
                with payload.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):
                        digest.update(block)
                after=payload.stat()
                if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or digest.hexdigest()!=row['sha256']:
                    raise ValueError('payload bytes changed or differ from recorded SHA256: '+name)
                rows_by_name[name].append(row)
            for name,rows in rows_by_name.items():
                rows.sort(key=lambda row:row['offset']);offset=0
                for index,row in enumerate(rows):
                    if row['offset']!=offset or row.get('batch_index')!=index:
                        raise ValueError('gapped or duplicate early tensor coverage: '+name)
                    offset+=row['input_tokens']
                if offset!=args.tokens:
                    raise IncompleteWindow('early window is not yet complete: '+name)
            return rows_by_name
        a,b=selected(args.reference),selected(args.candidate)
        checks=[{'name':name,**_token_compare(args.reference,args.candidate,a[name],b[name],name)} for name in sorted(EXPECTED_NAMES)]
        if any(check['tokens']!=args.tokens for check in checks):
            raise ValueError('stitched tensor token coverage differs from requested window')
        identical=all(check['passed'] for check in checks)
        result.update(complete_window=True,window_identical=identical,early_rejection=not identical,
            status='identical_window' if identical else 'diverged',checks=checks)
    except IncompleteWindow as error:
        result.update(status='incomplete_window',errors=[str(error)])
    except (OSError,ValueError,TypeError,KeyError) as error:
        result.update(status='invalid_capture',errors=[str(error)])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('repo','reference','candidate','output'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--tokens',type=int,required=True)
    args=p.parse_args();result=compare(args)
    temporary=args.output.with_suffix(args.output.suffix+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(args.output)
    print(json.dumps({key:value for key,value in result.items() if key!='checks'}))
    return 0 if result['status']=='identical_window' else 2

if __name__=='__main__':raise SystemExit(main())
