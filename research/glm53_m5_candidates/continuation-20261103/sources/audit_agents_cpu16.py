"""PREPARED SOURCE ONLY: offline C16 closure audit, no RONIN/native imports.

Do not execute while C16 is active. Reads only closed evidence, optionally copies
the closed SQLite DB/WAL/SHM to a temporary workspace before querying the copy.
Never reads API keys, process environments, model weights, or starts a process.
No model/numerical/performance/global qualification is granted by this audit.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

WORK=Path(__file__).resolve().parent
MAX_JSON=64<<20
TASKS={
 1:{'marker':'RONIN-ALPHA-713','prompt':"Un atelier prépare 17 lots de 24 pièces et dispose de 420 pièces. Calcule la quantité utilisée et la marge restante, puis vérifie si le stock suffit. Donne le calcul et une conclusion utile, en rappelant le repère privé de ta mémoire."},
 600:{'marker':'RONIN-OMEGA-962','prompt':"Pour une commande de 12 unités, le fournisseur A facture 35 € de frais fixes puis 8 € par unité ; B facture 10 € puis 11 € par unité. Compare les coûts, choisis le moins cher et calcule l'économie. Donne le calcul et ta justification, en rappelant le repère privé de ta mémoire."}}
# Exact reviewed nine-artifact set from this original C16 launch.
EXPECTED_METADATA_SHA256={
  "/Users/macbook/Documents/Codex/2026-11-02/referenced-chatgpt-conversation-this-is-an-2/work/runtime-v4/bin/llama-server": "a0c7632523ae199f576b8b63e423f0b0dc80068545a62d45d2132ceb765a9f83",
  "/Users/macbook/Documents/Codex/2026-11-02/referenced-chatgpt-conversation-this-is-an-2/work/runtime-v4/manifest.json": "7b0ac6dddabf542aae42d7fb8bb8762d6b98e1705dc423b09ad071112d0c8750",
  "/Users/macbook/Documents/Codex/2026-11-03/referenced-chatgpt-conversation-this-is-an/work/memory-recovery-20261103/campaign-3/model-verification.json": "6f72ce88dbad7292a57a6f715e43e60f89aa3ec4686d8278fea6710c3ba24d86",
  "/Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/airllm-next/air_llm/airllm/nextgen/glm_native.py": "a9d8a058e97c6cb47019df774e3e1a1c705da5a66879766ea1efe103c15a9ed0",
  "/Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/airllm-next/integrations/ronin/engine/adapter.py": "6a9e7b2aa61bd2d6d1b3d1aa0abf8e64d3164166cc7476cf49cc58eaaee3a218",
  "/Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/airllm-next/integrations/ronin/runtime/agent_context.py": "8babb6f477f154de547b99de073961c80be4f3e310083ec1d6dae5e0167385fa",
  "/Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/airllm-next/integrations/ronin/runtime/ronin_runtime.py": "a4bb3ee49aabccdb5500ca1adc5c95fc1d62dffa8d3a42e6e73740f614e17e51",
  "/Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/airllm-next/research/glm53_shared_prefill/qualify_agents.py": "7e22296df201904f2b45e3779b5e7f5f5c41acd1ccd5a9ec5d284364f4192b1f",
  "/Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/airllm-next/research/glm53_batched_prefill/campaign.py": "fdddb785578e9a5d491474cf5a1d6005351482015dc5c325554de53f4747934e"
}
EXPECTED_FLAGS={'--batch-size':'16','--ubatch-size':'16','--threads':'10','--gpu-layers':'0','--parallel':'1','--ctx-size':'2048','--device':'none','--cache-ram':'0','--ctx-checkpoints':'0','--cache-reuse':'0','--spec-type':'none','--fit':'off','--load-mode':'mmap'}
REQUIRED_FLAGS=('--offline','--no-repack','--no-warmup','--no-op-offload','--no-cache-idle-slots','--no-cache-prompt','--no-context-shift')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest_text(text):return hashlib.sha256(text.encode()).hexdigest()
def ident(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def bounded(path,cap=MAX_JSON):
    path=Path(path)
    with path.open('rb') as source:data=source.read(cap+1)
    if len(data)>cap:raise ValueError('bounded evidence exceeded: '+str(path))
    return data
def finite_float(value):
    parsed=float(value)
    if not math.isfinite(parsed):raise ValueError('nonfinite JSON float')
    return parsed
def read(path):return json.loads(bounded(path),parse_float=finite_float,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def safe(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('missing/escaping evidence: '+str(name))
    return path
def rows(path):
    data=bounded(path)
    if data and not data.endswith(b'\n'):raise ValueError('unclosed JSONL: '+str(path))
    return [json.loads(line,parse_float=finite_float,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite event JSON'))) for line in data.splitlines() if line.strip()]

def redact(value):
    # Exact text policy copied from the pinned RONIN source; no runtime import.
    value=str(value)
    value=re.sub(r'(?is)-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----','[clé privée expurgée]',value)
    value=re.sub(r'(?i)(authorization\s*[:=]\s*)(?:bearer\s+|basic\s+)?[^\s\"\']+',r'\1[expurgé]',value)
    value=re.sub(r'(?i)((?:api[_-]?key|access[_-]?token|secret|password)\s*[\"\']?\s*[:=]\s*[\"\']?)[^\s\"\',;]+',r'\1[expurgé]',value)
    return re.sub(r'\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{16,}|hf_[A-Za-z0-9]{16,})\b','[identifiant expurgé]',value)

def final_answer(text):return text.rsplit('</think>',1)[1].strip() if isinstance(text,str) and '</think>' in text else ''
def useful(agent,text):
    final=final_answer(text);marker=TASKS[agent]['marker'];other=TASKS[600 if agent==1 else 1]['marker']
    numbers=set(re.findall(r'(?<!\d)\d+(?!\d)',final));expected={'408','12'} if agent==1 else {'131','142','11'}
    clauses=re.split(r'[.!?;,\n]+',' '.join(final.casefold().split()))
    negated=re.compile(r"\b(?:ne|pas|jamais|aucun|aucune)\b|\bn['’]")
    affirmative=[clause for clause in clauses if not negated.search(clause)]
    if agent==1:
        stock=r'\bstock(?:\s+(?:actuel|disponible))?\s+(?:(?:est|reste|sera)\s+(?:donc\s+)?)?(?:suffit|suffisant)\b'
        contradicted=any('stock'in clause and(negated.search(clause)or'insuffisant'in clause)and re.search(r'suffi',clause)for clause in clauses)
        conclusion=not contradicted and any(re.search(stock,clause)for clause in affirmative)
    else:
        def chooses(supplier,clause):
            label=r'\b'+supplier+r'\b'
            cheaper=label+r'\s*(?:est|reste|revient|:)\s*(?:donc\s+)?(?:le\s+)?(?:moins\s+cher|moins\s+coûteux|plus\s+économique)\b'
            choice=r'\b(?:chois(?:is|issons|it)|reti(?:ens|ent)|retenons|privilégi(?:e|ons)|recommand(?:e|ons))\s+(?:donc\s+)?(?:(?:le\s+)?fournisseur\s+)?'+label
            return bool(re.search(cheaper,clause)or re.search(choice,clause))
        conclusion=any(chooses('a',clause)for clause in affirmative)and not any(chooses('b',clause)for clause in affirmative)
    return {'nonempty_final_answer':bool(final),'own_marker_in_final':marker in final,'peer_marker_absent':other not in text,
        'expected_result_numbers':expected<=numbers,'visible_calculation_or_justification':bool(re.search(r'[=×*]|(?:calcul|car |donc|reste|suffi|économ|moins cher|marge)',final,re.I)),
        'correct_conclusion':bool(conclusion)}

def completion(row,events):
    final=row.get('final')or{};tokens=row.get('tokens');body=row.get('body')or{}
    valid_ids=isinstance(tokens,list)and bool(tokens)and all(type(t)is int and t>=0 for t in tokens)
    flattened=[token for event in events for token in event['response'].get('tokens',[])]
    terminals=[event['response'] for event in events if event['response'].get('stop')]
    count=final.get('tokens_predicted')
    return {'planned512':type(body.get('n_predict'))is int and body['n_predict']==512,
        'one_exact_terminal':len(terminals)==1 and terminals[0]==final,'natural_eos':final.get('stop')is True and final.get('stop_type')=='eos',
        'not_truncated':final.get('truncated')is False,'complete_generated_ids':valid_ids and type(count)is int and 0<count<=512 and len(tokens)==count and flattened==tokens,
        'timing_count_agrees':'predicted_n'not in(final.get('timings')or{})or type(final['timings']['predicted_n'])is int and final['timings']['predicted_n']==count,
        'raw_text_agrees':''.join(event['response'].get('content','')for event in events)==row.get('text')}

def first_two_observation(run):
    """Descriptive raw prefix only; never replaces the full audit or provenance."""
    run=Path(run).resolve();report=read(safe(run,'report.json'))
    if report.get('status')=='running' or report.get('owned_server_stopped')is not True:return None
    requests=report.get('requests')or[]
    if len(requests)<2 or [r.get('agent_id')for r in requests[:2]]!=[1,600]:return False
    for index,row in enumerate(requests[:2]):
        agent=row['agent_id'];event_path=safe(run,f'request-{index+1:02d}-agent-{agent}/agent-{agent}-events.jsonl')
        events=rows(event_path)
        if events!=row.get('events')or not all(completion(row,events).values())or not all(useful(agent,row.get('text','')).values()):return False
    return True

def closed_database(root,scratch,tracked):
    db=safe(root,'state/ronin.sqlite');names=['ronin.sqlite']
    for suffix in ('-wal','-shm'):
        if (db.parent/(db.name+suffix)).is_file():names.append(db.name+suffix)
    total=0
    for name in names:
        source=db.parent/name;size=source.stat().st_size;total+=size
        if total>128<<20:raise ValueError('bounded SQLite snapshot exceeded')
        tracked[str(source)]=ident(source);shutil.copyfile(source,scratch/name)
        if ident(source)!=tracked[str(source)]:raise ValueError('closed SQLite evidence changed during copy')
    # Locks/recovery, if any, affect only this temporary copy. Original evidence
    # is never opened by SQLite and no original WAL checkpoint can be triggered.
    connection=sqlite3.connect((scratch/'ronin.sqlite').as_uri()+'?mode=ro',uri=True)
    connection.execute('PRAGMA query_only=ON');connection.row_factory=sqlite3.Row
    try:
        identities=[r['agent_id']for r in connection.execute('SELECT agent_id FROM glm_agent_identities ORDER BY agent_id')]
        decisions=[dict(r)for r in connection.execute('SELECT * FROM glm_agent_decisions ORDER BY sequence')]
        contexts=[dict(r)for r in connection.execute('SELECT agent_id,mission_id,working_memory,revision FROM glm_agent_contexts WHERE agent_id IN(1,600)')]
        for row in decisions:
            for key in ('tokens','messages','provenance','terminal'):row[key]=json.loads(row[key])if row[key]is not None else None
        return identities,decisions,contexts
    finally:connection.close()

def audit(run,observation):
    run=Path(run).resolve();report_path=safe(run,'report.json');report=read(report_path)
    result={'schema_version':1,'status':'not_qualified','checks':{},'requests':[],'errors':[],
        'C_global_qualification':False,'numerical_equivalence_qualified':False,'performance_qualified':False,'all600_generated':False,
        'first_two_useful_eos_observed':None,'full_sequence_history_audited':None,
        'raw_final_answer_review_required':True,'scope':'independent closed C16 functional pilot1→600→1 and exact private-history replay; no A/B or global proof'}
    # Refuse live DB/SSE reads before any evidence processing beyond the report.
    if report.get('status')=='running'or report.get('owned_server_stopped')is not True:
        result.update(status='not_closed',errors=['C16 is live or owned process closure is unconfirmed; no DB/SSE inspected']);return result
    tracked={str(report_path):ident(report_path)};checks=result['checks']
    checks['reported_completed']=report.get('status')=='completed'
    checks['exact_agents_mode']=report.get('mode')=='agents'
    checks['native_or_max_requested']=('reasoning_effort'in report and report['reasoning_effort']in(None,'max'))
    checks['planned_sequence']=[r.get('agent_id')for r in report.get('planned_agents',[])]==[1,600,1]
    elapsed=report.get('elapsed_seconds')
    checks['observed_lifecycle_duration_finite']=type(elapsed)in(int,float)and math.isfinite(elapsed)and elapsed>0
    result['duration_observation']={'lifecycle_seconds_including_cleanup':elapsed,'lifecycle_within7200':checks['observed_lifecycle_duration_finite']and elapsed<=7200,'exact_pre_cleanup_monitored_seconds':None,'global_deadline_scope':'original monitor is authoritative; report.elapsed_seconds includes cleanup, so do not equate it to the exact monitored interval'}
    checks['budgets']=report.get('requested_tokens_each')==512 and report.get('explicit_request_budget_seconds')==2700 and report.get('max_campaign_seconds')==7200
    checks['normal_native_exit']=type(report.get('owned_server_returncode'))is int and report['owned_server_returncode']==0
    checks['no_cleanup_or_supervision_error']=not any(report.get(k)for k in('cleanup_errors','cleanup_error','supervision_errors','error_code','final_power_error'))
    checks['original_functional_checks']=bool(report.get('functional_checks'))and all(value is True for value in report['functional_checks'].values())
    profile=report.get('profile')or{}
    checks['actual16_profile']=profile.get('microbatch')==16 and profile.get('logical_batch')==16 and profile.get('gpu_layers')==0 and profile.get('cache_shared')is False
    admission=report.get('batch_admission')or{}
    checks['actual16_admitted']=admission.get('admitted')is True and admission.get('selected_microbatch')==16 and(admission.get('sample')or{}).get('pressure_level')==1
    config_path=safe(run,'run-config.json');tracked[str(config_path)]=ident(config_path);config=read(config_path)
    checks['hardcaps_preserved']=config.get('limits',{}).get('max_rss_bytes')==int(8.5*(1<<30)) and config['limits'].get('max_swap_growth_bytes')==32<<20 and config['limits'].get('warning_swap_growth_bytes')==32<<20
    checks['reserve_policy_preserved']=config['limits'].get('reserve_bytes',0)>=768<<20 and config['limits'].get('warning_headroom_bytes',0)>=512<<20 and 0<config['limits'].get('headroom_grace_seconds',0)<=3 and 0<config['limits'].get('warning_grace_seconds',0)<=20
    observed=report.get('observed_native_argv')or{};argv=observed.get('argv')or[]
    checks['OS_argv_exact']=observed.get('available')is True and observed.get('pid')==report.get('owned_pid') and argv==(report.get('constructed_native_command')or{}).get('argv')
    checks['OS_args_CPU16_no_cache']=all(argv.count(flag)==1 and argv.index(flag)+1<len(argv)and argv[argv.index(flag)+1]==value for flag,value in EXPECTED_FLAGS.items())and all(flag in argv for flag in REQUIRED_FLAGS)
    pinned_runtime=next(path for path in EXPECTED_METADATA_SHA256 if path.endswith('/bin/llama-server'))
    if (config.get('runtime_binary')!=pinned_runtime or not argv or argv[0]!=pinned_runtime
            or config.get('expected_runtime_sha256')!=EXPECTED_METADATA_SHA256[pinned_runtime]):
        raise ValueError('runtime path/SHA must be the exact pinned executable before any runtime hash read')
    checks['actual_runtime_file']=sha(pinned_runtime)==EXPECTED_METADATA_SHA256[pinned_runtime]
    sources=report.get('metadata_preflight',{}).get('sha256',{})
    if (type(sources)is not dict or sources!=EXPECTED_METADATA_SHA256 or
            any(type(digest)is not str or re.fullmatch('[0-9a-f]{64}',digest)is None for digest in sources.values())):
        raise ValueError('complete exact reviewed nine-artifact metadata required; no declared arbitrary file is read')
    checks['complete_pinned_metadata']=True
    for name,digest in EXPECTED_METADATA_SHA256.items():
        tracked[name]=ident(name)
        checks['source:'+name]=Path(name).is_file()and sha(name)==digest
    requests=report.get('requests')or[]
    checks['three_observed_requests']=len(requests)==3 and [r.get('agent_id')for r in requests]==[1,600,1]
    checks['same_declared_pid']=len(requests)==3 and all(r.get('owned_pid')==report.get('owned_pid')for r in requests)
    native=report.get('engine_after_stream',{}).get('native',{});after=report.get('engine_after_cleanup',{}).get('native',{})
    checks['one_real_load_three_requests']=native.get('counters',{}).get('loads')==1 and native.get('counters',{}).get('requests')==3
    checks['one_physical_slot']=native.get('scheduler',{}).get('physical_limit')==1
    checks['native_guards_authoritative_clear']=bool(native)and bool(after)and native.get('failure')is None and after.get('failure')is None
    checks['weights_runtime_verified']=native.get('runtime_artifacts_verified')is True and native.get('checkpoint',{}).get('integrity_verified')is True and not native.get('checkpoint',{}).get('errors')
    checks['power_receipt_clear']=report.get('power_continuity_validated')is True and report.get('power_gate',{}).get('passed')is True and report.get('power_gate',{}).get('finalized')is True and report.get('power_gate',{}).get('violation')is None
    sample_path=safe(run,'native-resource-samples.json');tracked[str(sample_path)]=ident(sample_path);samples=read(sample_path)
    valid_samples=isinstance(samples,list)and bool(samples);previous=None
    if valid_samples:
        for sample in samples:
            moment=sample.get('monotonic')
            if (type(moment)not in(int,float)or not math.isfinite(moment)or moment<0 or(previous is not None and moment<=previous)
                    or any(type(sample.get(key))is not int or sample[key]<0 for key in('rss_bytes','available_bytes','swap_used_bytes','pressure_level'))
                    or sample['rss_bytes']>int(8.5*(1<<30))or sample['pressure_level']not in(1,2)):
                valid_samples=False;break
            previous=moment
    checks['native_resource_sample_hardcaps_finite']=valid_samples
    reported_peak=native.get('peak_sampled_rss_bytes')
    checks['reported_sampled_RSS_cap']=type(reported_peak)is int and 0<=reported_peak<=int(8.5*(1<<30))and valid_samples and reported_peak>=max(s['rss_bytes']for s in samples)
    result['resource_samples']={'retained_count':len(samples)if isinstance(samples,list)else None,
        'bounded_deque_capacity_from_source':3600,'complete_campaign_sample_history_guaranteed':False,
        'scope':'original bounded sampled history and maintained RSS peak; no unsampled-peak or invented swap baseline proof'}
    rows_by_id={}
    with tempfile.TemporaryDirectory(prefix='c16-audit-',dir=run.parent)as temporary:
        identities,decisions,contexts=closed_database(run,Path(temporary),tracked)
    checks['exact600_identity_ids']=identities==list(range(1,601))and all(type(i)is int for i in identities)
    checks['DB_decision_order']=[r.get('agent_id')for r in decisions]==[1,600,1]and len(decisions)==3
    for index,row in enumerate(requests):
        agent=row.get('agent_id')
        if agent not in TASKS:raise ValueError('unexpected agent outside observed pilot')
        event_name=f'request-{index+1:02d}-agent-{agent}/agent-{agent}-events.jsonl'
        event_path=(run/event_name).resolve()
        if not event_path.is_relative_to(run):raise ValueError('escaping request event path')
        available=event_path.is_file()
        if available:
            tracked[str(event_path)]=ident(event_path);events=rows(event_path)
        else:
            events=[]  # Missing failed-third transcript is a failed check, never invented.
        body=row.get('body')or{};decision=row.get('decision')or{};messages=decision.get('messages')or[]
        complete=completion(row,events);peer=TASKS[600 if agent==1 else 1]['marker'];own=TASKS[agent]['marker']
        check={'request_index':index,'agent_id':agent,'completion':complete,'useful':useful(agent,row.get('text',''))}
        check['raw_event_file_available']=available
        check['events_equal_report']=events==row.get('events')
        check['template_messages_exact']=(row.get('template_request')or{}).get('messages')==messages
        kwargs=(row.get('template_request')or{}).get('chat_template_kwargs')
        check['template_kwargs_native_or_max']=kwargs is None or kwargs=={'reasoning_effort':'max'}
        rendered=row.get('rendered_prompt')or'';system_prefix=rendered.split('<|user|>',1)[0]
        check['native_max_rendered']='<|system|>Reasoning Effort: Max'in system_prefix and not re.search(r'Reasoning Effort: (?:Low|High)',system_prefix)
        check['native_render_sent_unmodified']=body.get('prompt')==rendered and(row.get('tokenize_request')or{}).get('content')==rendered and(row.get('tokenize_request')or{}).get('add_special')is True
        ids=row.get('prompt_token_ids');check['actual_native_prompt_ids']=isinstance(ids,list)and bool(ids)and all(type(t)is int and t>=0 for t in ids)
        check['cache_disabled']=body.get('cache_prompt')is False and type(body.get('id_slot'))is int and body['id_slot']==0
        expected_body={'prompt':rendered,'n_predict':512,'stream':True,'temperature':0,'seed':42,'cache_prompt':False,'id_slot':0,
            'return_tokens':True,'n_probs':0,'post_sampling_probs':False,'timings_per_token':True,'sse_ping_interval':1}
        check['native_payload_no_forced_stop_or_reasoning_shortcut']=body==expected_body
        check['native_context_bound']=check['actual_native_prompt_ids']and len(ids)+512<=2048
        check['markers_isolated']=own in rendered and peer not in rendered and own in canonical(messages)and peer not in canonical(messages)and peer not in row.get('text','')
        check['native_result_exact']=(row.get('result')or{}).get('text')==row.get('text')and(row.get('result')or{}).get('tokens')==row.get('tokens')
        db=[d for d in decisions if d['id']==decision.get('id')]
        check['DB_decision_matches_snapshot']=len(db)==1 and all(db[0].get(key)==decision.get(key)for key in('id','agent_id','mission_id','state','prompt','output_text','tokens','messages','context_sha256','memory_revision','terminal'))
        check['decision_complete_eos']=decision.get('state')=='completed'and(decision.get('terminal')or{}).get('stop_type')=='eos'
        check['decision_tokens_exact']=decision.get('tokens')==row.get('tokens')
        check['stored_text_exact_redaction']=decision.get('output_text')==redact(row.get('text',''))
        check['context_digest_exact']=decision.get('context_sha256')==digest_text(canonical(messages))
        check['full_user_task']=decision.get('prompt')==TASKS[agent]['prompt']and bool(messages)and messages[-1]=={'role':'user','content':TASKS[agent]['prompt']}
        engine=(row.get('engine_after')or{}).get('native',{})
        check['native_load_request_slot_counts']=engine.get('counters',{}).get('loads')==1 and engine.get('counters',{}).get('requests')==index+1 and engine.get('scheduler',{}).get('physical_limit')==1
        check['per_request_clear_guard']=engine.get('failure')is None and bool(engine)
        native_seconds=(engine.get('last_request')or{}).get('seconds')
        runtime_seconds=(row.get('result')or{}).get('seconds');wrapper_seconds=row.get('wall_seconds')
        finite=lambda value:type(value)in(int,float)and math.isfinite(value)and value>0
        check['observed_native_request_budget']=finite(native_seconds)and native_seconds<=2700
        check['broader_recorded_durations_finite']=finite(runtime_seconds)and finite(wrapper_seconds)
        duration={'native_stream_seconds':native_seconds,'runtime_generate_seconds_including_context_work':runtime_seconds,
            'capture_wrapper_wall_seconds':wrapper_seconds,'scope':'only native last_request.seconds is compared with the native2700s request budget; broader timings include other work'}
        # No raw model text/prompts are copied to the audit output.
        result['requests'].append({**check,'raw_text_sha256':digest_text(row.get('text','')),'stored_text_sha256':digest_text(decision.get('output_text','')),'prompt_tokens':len(ids)if isinstance(ids,list)else None,'generated_tokens':len(row.get('tokens',[])),'duration_observation':duration})
        rows_by_id[index]=row
    checks['same_mission_on_return']=len(requests)==3 and requests[0].get('mission_id')==requests[2].get('mission_id')and requests[0].get('mission_id')!=requests[1].get('mission_id')
    if len(requests)==3:
        first=requests[0].get('decision')or{};third=requests[2].get('decision')or{};m=third.get('messages')or[]
        pair=[{'role':'user','content':first.get('prompt')},{'role':'assistant','content':first.get('output_text')}]
        checks['exact_first_pair_replayed']=bool(first)and bool(third)and len(m)==4 and m[1:3]==pair and m[-1]=={'role':'user','content':TASKS[1]['prompt']}
        checks['first_context_fresh']=len(first.get('messages')or[])==2
        checks['peer_context_fresh']=len((requests[1].get('decision')or{}).get('messages')or[])==2
        memory=[c for c in contexts if c['agent_id']==1 and c['mission_id']==third.get('mission_id')]
        checks['return_memory_isolated']=len(memory)==1 and TASKS[1]['marker']in memory[0]['working_memory']and TASKS[600]['marker']not in memory[0]['working_memory']
    else:
        checks.update(exact_first_pair_replayed=False,first_context_fresh=False,peer_context_fresh=False,return_memory_isolated=False)
    observation=Path(observation)
    if observation.is_file():
        tracked[str(observation)]=ident(observation);late=read(observation)
        locks=late.get('native_open_uid_locks')or[];parents=late.get('parent_open_uid_locks')or[]
        lock_matches=(len(locks)==1 and all(type(lock.get('fd'))is int and lock['fd']>=3
            and type(lock.get('device'))is int and type(lock.get('inode'))is int and lock['inode']>0
            and str(lock.get('path','')).endswith('/f7ed7077cbc8482dac0d3711.lock')
            and any(all(parent.get(key)==lock.get(key)for key in('path','device','inode'))for parent in parents)for lock in locks))
        checks['late_UID_observation_matches']=late.get('pid')==report.get('owned_pid')and late.get('create_time')==observed.get('create_time')and late.get('executable')==observed.get('executable')and late.get('same_uid_lock_inode_open_in_native_and_parent')is True and late.get('os_argv_matches_constructed')is True and lock_matches
        result['passive_UID_observation']={'path':str(observation.resolve()),'sha256':sha(observation),'timestamp_utc':late.get('timestamp_utc'),'scope':late.get('ownership_scope'),'before_GO_attested':False}
    else:checks['late_UID_observation_matches']=False
    for path,value in tracked.items():
        if ident(path)!=value:raise ValueError('closed evidence changed during audit: '+path)
    def all_checks(value):
        return all(all_checks(v)if isinstance(v,dict)else v is True for v in value.values())
    per_request=all(all_checks({k:v for k,v in row.items()if k not in('request_index','agent_id','raw_text_sha256','stored_text_sha256','prompt_tokens','generated_tokens','duration_observation')})for row in result['requests'])
    passed=bool(requests)and all_checks(checks)and per_request
    first_two=result['requests'][:2]
    result['first_two_useful_eos_observed']=(len(first_two)==2 and [r['agent_id']for r in first_two]==[1,600]
        and all(all_checks(r['completion'])and all_checks(r['useful'])and r['events_equal_report'] for r in first_two))
    result['full_sequence_history_audited']=(len(requests)==3 and checks['DB_decision_order']and checks['same_mission_on_return']
        and checks['exact_first_pair_replayed']and checks['first_context_fresh']and checks['peer_context_fresh']and checks['return_memory_isolated']
        and all(r['DB_decision_matches_snapshot']and r['context_digest_exact']and r['template_messages_exact']and r['markers_isolated']and r['decision_complete_eos'] for r in result['requests']))
    result.update(status='closed_evidence_passed'if passed else'closed_evidence_rejected',functional_contract_and_history_passed=passed,
        source_sha256=sha(__file__),report_sha256=sha(report_path),closed_inputs_sha256={path:sha(path)for path in tracked},
        distinct_agents_observed=len({r.get('agent_id')for r in requests}),
        resource_scope={'original_native_guard_attested_clear':checks['native_guards_authoritative_clear'],
            'independent_exact_native_baseline_replay':False,'reason':'Native _resource_baseline is not explicitly persisted by the original campaign; do not invent it from the first sample or conflate it with batch_admission',
            'peak_global_swap_growth_from_original_native_baseline_bytes':None},
        limits={'scope':'sampled authoritative original watchdog, not proof of unsampled physical peaks'},
        raw_answer_review='Review the three raw final answers in the private source evidence; heuristic useful checks are not a universal semantic proof')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=WORK/'agents-cpu16-native')
    parser.add_argument('--live-observation',type=Path,default=WORK/'agents-cpu16-live-observation.json')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    try:value=audit(args.run,args.live_observation)
    except (OSError,ValueError,KeyError,TypeError,AttributeError,sqlite3.Error)as error:
        try:prefix=first_two_observation(args.run)
        except (OSError,ValueError,KeyError,TypeError,AttributeError):prefix=None
        value={'status':'invalid_closed_evidence','functional_contract_and_history_passed':False,'C_global_qualification':False,
            'first_two_useful_eos_observed':prefix,'full_sequence_history_audited':False,
            'prefix_scope':'descriptive closed raw SSE/usefulness only; does not repair invalid full evidence or establish global/native provenance',
            'errors':[str(error)]}
    args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return 0 if value.get('status')=='closed_evidence_passed' else 2

if __name__=='__main__':raise SystemExit(main())
