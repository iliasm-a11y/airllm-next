"""Read-only checks for a stale M5 index after an interrupted atomic state save.

This authorizes reindexing recorded paths, never accepting a worker. Unknown
paths and operations that died before recording ownership remain fail-closed.
The caller holds the campaign lock, checks frozen inputs, and cleans owned
leases before replacing an index or resuming execution.
"""
import math
from pathlib import Path

from verify_m5_validation import (VerifiedFidelityRejection,
    digest, inside, read_json, require, summarize_experiment, validate_job,
    validate_kernel_report, verify_hash, verify_manifest, finder_metadata_ignored)


RECOVERABLE = frozenset({'running','deferred_or_interrupted',
    'partial_selected_experiments_completed','completed','completed_with_exclusions',
    'completed_evidence_with_rejections'})
KNOWN = RECOVERABLE | {'prepared_no_worker_launched'}
COMMON = frozenset({'cleanup.json','controller-child.log','owned-processes.json'})


def inspect_index(folder,config,state):
    """Check the old immutable hashes and every newly present path; no writes."""
    folder=Path(folder).resolve(strict=True)
    require(config.get('origin')==str(folder),'Recovery must use the original campaign folder')
    require(state.get('status') in KNOWN,'Unknown campaign recovery state')
    require(read_json(folder/'config.json') == config and read_json(folder/'state.json') == state,
            'Campaign state/config changed during index inspection')
    require(state.get('config_sha256') == digest(folder/'config.json'),'Campaign configuration identity changed')
    paths=[path for path in folder.rglob('*') if not finder_metadata_ignored(path)]
    for path in paths:
        require(not path.is_symlink(),'Campaign recovery cannot contain symlinks')
        require(path.is_file() or path.is_dir(),'Unsupported campaign recovery filesystem entry')
        if path.is_file() and path.name in ('manifest.json','.runner.lock'):
            require(path.parent==folder,'Reserved index/lock name outside campaign root')
    previous=read_json(folder/'manifest.json')
    files=previous.get('files',{})
    require({'config.json','state.json'} <= set(files),'Manifest lacks frozen campaign inputs')
    recovering=state['status'] in RECOVERABLE
    verify_manifest(folder,allow_interrupted=recovering)
    actual={str(p.relative_to(folder)) for p in paths if p.is_file()
            and p.name not in ('manifest.json','.runner.lock')}
    added=actual-set(files)
    state_changed=(files['state.json']['bytes'] != (folder/'state.json').stat().st_size
                   or files['state.json']['sha256'] != digest(folder/'state.json'))
    result=dict(needed=bool(added or state_changed),added_files=sorted(added),
                manifest_sha256=digest(folder/'manifest.json'),state_sha256=digest(folder/'state.json'),
                config_sha256=digest(folder/'config.json'))
    if not result['needed']:
        verify_manifest(folder)
        return result
    require(recovering,'Campaign state does not permit index recovery')
    plans={experiment['name']+'/'+job['id']:job for experiment in config['experiments'] for job in experiment['plan']}
    sessions=state.get('sessions',[])
    require(isinstance(sessions,list),'Invalid recorded recovery sessions')
    allowed={}
    def add_names(parent,names):
        allowed.setdefault(parent,set()).update(names)
    def relative(raw):
        path=Path(raw)
        if path.is_absolute():
            require(path.is_relative_to(folder),'Recorded recovery path escapes campaign')
            path=path.relative_to(folder)
        require('..' not in path.parts and not path.is_absolute(),'Unsafe recorded recovery path')
        return path
    def job_names(key,record,index):
        require(key in plans,'Recovery worker is outside frozen plan')
        require(type(index) is int and 0 <= index < len(sessions),'Invalid recovery worker session')
        expected=Path('sessions')/f'{index:04d}'/key/'worker.json'
        require(relative(record['output']) == expected,'Recorded recovery worker output differs from frozen plan')
        if record.get('output_absolute') is not None:
            require(relative(record['output_absolute']) == expected,'Recovery absolute worker path differs')
        tasks=config['tasks_by_path'][plans[key]['tasks']]
        require(isinstance(tasks,list) and len(tasks)==1,'Recovery requires one frozen recording task')
        count=tasks[0]['max_new_tokens']
        require(type(count) is int and 0 < count <= 4096,'Unbounded recovery artifact count')
        flags=plans[key]['args']
        speculative=('--speculative-depth' in flags and flags[flags.index('--speculative-depth')+1] != '0')
        count=count*4+1 if speculative else count
        add_names(expected.parent,COMMON|{'worker.json','worker.guard.json','worker.log'}|
                  {f'worker.step-{i:03d}.f32' for i in range(count)})
    for ordinal,session in enumerate(sessions):
        require(type(session.get('index')) is int and session['index']==ordinal,'Invalid recovery session identity')
        parent=Path('sessions')/f'{ordinal:04d}'
        hash_paths=[session[name] for name in ('hash_before','hash_after') if name in session]
        for block in session.get('integrity_blocks',[]):
            require(type(block.get('index')) is int and block['index'] >= 0,'Invalid recorded hash-block identity')
            hash_paths.extend(block[name] for name in ('hash_before','hash_after') if name in block)
        for raw in hash_paths:
            path=relative(raw)
            require(path.name=='checkpoint.json' and path.parent.parent==parent
                    and (path.parent.name in ('hash-before','hash-after')
                         or any(path.parent.name==f'integrity-{block["index"]:04d}'
                                for block in session.get('integrity_blocks',[]))),
                    'Unowned recovery checkpoint-hash path')
            add_names(path.parent,COMMON|{'checkpoint.json','checkpoint.guard.json','checkpoint.log'})
        if session.get('kernel_report') is not None:
            path=relative(session['kernel_report'])
            require(path==parent/'kernels/kernels.json','Unowned recovery kernel-report path')
            add_names(path.parent,COMMON|{'kernels.json','kernels.guard.json','kernels.log'})
            report=read_json(inside(folder,str(path)))
            for case in report.get('cases',[]):
                descriptors=list(case.get('numeric_evidence',{}).values())
                if case.get('candidate_evidence') is not None:
                    descriptors.append(case['candidate_evidence'])
                for descriptor in descriptors:
                    raw=Path(descriptor['file'])
                    require(raw.parent==Path('kernels.tensors') and raw.name not in ('','..'),
                            'Unsafe recorded recovery kernel tensor')
                    add_names(path.parent/raw.parent,{raw.name})
        active=session.get('active')
        if active:
            require(active.get('session',ordinal)==ordinal,'Active recovery session differs')
            job_names(active['experiment']+'/'+active['job'],active,ordinal)
    records=[state.get('jobs',{})]+state.get('discarded_unbracketed_jobs',[])
    for bank in records:
        require(isinstance(bank,dict),'Invalid recorded recovery jobs')
        for key,record in bank.items():
            job_names(key,record,record['session'])
    for name in added:
        path=Path(name)
        require(path.name in allowed.get(path.parent,set()),'Unlisted file outside recorded campaign ownership: '+name)
        inside(folder,name)
    return result


def validate_recovery_records(folder,config,state):
    """Recheck recorded sealed claims with the unchanged independent validators."""
    plans={experiment['name']+'/'+job['id']:job for experiment in config['experiments'] for job in experiment['plan']}
    sealed={}
    for session in state.get('sessions',[]):
        for name in ('hash_before','hash_after'):
            if name in session:
                verify_hash(folder,session[name],config)
        if session.get('kernel_report') is not None:
            validate_kernel_report(session['kernel_report'],config,folder)
        for block in session.get('integrity_blocks',[]):
            if block.get('status')!='sealed':
                continue
            identity=(session['index'],block['index'])
            require(identity not in sealed,'Duplicate recovery integrity block')
            verify_hash(folder,block['hash_before'],config)
            verify_hash(folder,block['hash_after'],config)
            sealed[identity]=block
    for key,record in state.get('jobs',{}).items():
        block=sealed.get((record['session'],record.get('integrity_block')))
        # Unsealed completed workers are retained as raw evidence and the
        # existing runner discards/reruns them; recovery never adds a seal.
        if block is None:
            continue
        require(key in block.get('jobs',[]),'Recovery worker is absent from its integrity block')
        bounds=(block.get('hash_before_completed_unix'),record.get('started_unix'),
                record.get('ended_unix'),block.get('hash_after_started_unix'))
        require(all(type(value) in (int,float) and math.isfinite(value) for value in bounds)
                and bounds[0] <= bounds[1] <= bounds[2] <= bounds[3],
                'Recovery worker is outside its recorded hash bracket')
        validate_job(folder,config,plans[key],record)
    for experiment in config['experiments']:
        recorded=state.get('experiment_results',{}).get(experiment['name'],{})
        if recorded.get('status') not in ('accepted','rejected_fidelity'):
            continue
        for job in experiment['plan']:
            key=experiment['name']+'/'+job['id']
            record=state.get('jobs',{}).get(key,{})
            block=sealed.get((record.get('session'),record.get('integrity_block')))
            require(block is not None and key in block.get('jobs',[]),'Recovered result lacks sealed workers')
        try:
            result=summarize_experiment(folder,config,experiment,state['jobs'])
        except VerifiedFidelityRejection as error:
            result=error.result
        require(result==recorded,'Recovered result differs from independent recalculation')
