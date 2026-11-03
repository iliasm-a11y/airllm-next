"""Generate 600 full research identities and recheck with native template/IDs.

Fresh isolated research missions; no personal application database is read.
Native default reasoning is preserved (Max in the pinned GLM template).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'air_llm')]
from research.glm53_shared_prefill.prefix_audit import generated_messages, common_prefix
from research.glm53_shared_prefill.qualify_agents import TASKS, metadata_preflight, configuration


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    config,_=configuration(args.config,512)
    evidence=metadata_preflight(config)
    if not evidence['checkpoint']['integrity_verified']:
        raise ValueError('unchanged model SHA receipts are required')
    args.output.mkdir(parents=True,exist_ok=False)
    messages=generated_messages(range(1,601),{task['agent_id']:task for task in TASKS})
    input_path=args.output/'messages.json'
    input_path.write_text(json.dumps({'agents':messages},ensure_ascii=False)+'\n')
    # Keep the previous metadata-only helper, replacing only its hardcoded low
    # reasoning override. No compute context or tensor shard is loaded.
    original=REPO/'research/glm53_shared_prefill/native_prefix.cpp'
    text=original.read_text()
    old='            request.chat_template_kwargs["reasoning_effort"] = "\\\"low\\\"";'
    if text.count(old)!=1:
        raise ValueError('native helper source contract changed')
    source=args.output/'native_prompts.cpp'
    source.write_text(text.replace(old,'            // Preserve native default reasoning; no template override.'))
    binary=args.output/'native-prompts'
    runtime=Path(config['runtime_binary']).parent
    command=['clang++','-std=c++17','-O2']
    for name in ('include','common','ggml/include','vendor'):
        command+=['-I',str(args.source/name)]
    command += [str(source),'-L',str(runtime),'-Wl,-rpath,'+str(runtime),'-lllama','-lllama-common','-o',str(binary)]
    subprocess.run(command,check=True,capture_output=True,timeout=60)
    output=args.output/'native-prompts.json'
    with output.open('wb') as stdout,(args.output/'native-stderr.log').open('wb') as stderr:
        completed=subprocess.run([str(binary),str(Path(config['model_path'])/'GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf'),str(input_path)],
            stdout=stdout,stderr=stderr,timeout=120)
    if completed.returncode:
        raise RuntimeError('native metadata audit failed')
    native=json.loads(output.read_text())
    prefix=common_prefix(a['token_ids'] for a in native['agents'])
    template=native['template']
    pair=[a for a in native['agents'] if a['agent_id'] in (1,600)]
    report={'complete':len(native['agents'])==600,'native_common_tokens':len(prefix),'prefix_ids':prefix,
        'candidate_boundary':208 if len(prefix)>=208 else None,'reasoning_effort_override':None,
        'native_template_sha256':hashlib.sha256(template.encode()).hexdigest(),
        'prompt_tokens_1_600':{str(a['agent_id']):len(a['token_ids']) for a in pair},
        'build_command':command,'binary_sha256':sha(binary),'source_sha256':sha(source),
        'original_source_sha256':sha(original),'metadata_preflight':evidence,
        'scope':'native metadata-only tokenization of 600 fresh full research identities; no prefill, snapshot or personal historical-session qualification'}
    for left,right in zip(messages,native['agents']):
        if left['agent_id']!=right['agent_id'] or left['marker'] not in right['rendered_prompt']:
            raise ValueError('identity or marker lost in full native prompt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ('complete','native_common_tokens','candidate_boundary','prompt_tokens_1_600')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('config','source','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
