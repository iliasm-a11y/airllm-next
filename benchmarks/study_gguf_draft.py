#!/usr/bin/env python3
"""Causal prompt-draft yield on recorded real tokens; NOT a speed benchmark.

The proposer sees only the prompt and already accepted prefix. Future target
tokens score its proposals afterwards. No target verification or recurrent
rollback is timed; no generation acceleration is inferred from oracle rounds.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter,process_time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'air_llm'))
from airllm.nextgen.prompt_draft import PromptSpanDraft


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();results=[]
    for file in sorted(a.campaign.glob('*-r0-reference.json')):
        t=json.loads(file.read_text())['turns'][0]
        for depth in [2,4]:
            draft=PromptSpanDraft();prefix=list(t['full_input_ids']);target=t['generated_ids'];position=0
            accepted=proposed=rounds=nonempty=0;wall=cpu=0.
            while position<len(target):
                start=perf_counter();c=process_time();proposal=draft(prefix,depth);cpu+=process_time()-c;wall+=perf_counter()-start
                proposed+=len(proposal);rounds+=1;nonempty+=bool(proposal);match=0
                for actual,expected in zip(proposal,target[position:]):
                    if actual!=expected:break
                    match+=1
                accepted+=match;advance=min(match+1,len(target)-position)
                prefix.extend(target[position:position+advance]);position+=advance
            results.append({'id':t['id'],'depth':depth,'target_tokens':len(target),'proposal_rounds':rounds,
                'nonempty_rounds':nonempty,'proposed_tokens':proposed,'matching_draft_tokens':accepted,
                'draft_wall_seconds':wall,'draft_cpu_seconds':cpu})
    result={'scope':'Causal low-cost proposal study on real recorded tokens; oracle scoring only, no target block execution',
        'generation_speedup_measured':False,'recurrent_rollback_cost_measured':False,'results':results}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
