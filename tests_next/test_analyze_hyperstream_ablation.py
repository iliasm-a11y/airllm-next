"""Fabricated four-worker evidence; no model, MLX or real timing measurement."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import analyze_hyperstream_ablation as analyzer
from test_analyze_hyperstream import hyperstream_fixture
from test_analyze_research51 import save


def fixture(folder,head=analyzer.EXPECTED_HEAD):
    original=hyperstream_fixture(folder);campaign=copy.deepcopy(original)
    root=original['plan'][0]['engine_root'];manifest=copy.deepcopy(original['engine_manifests'][root])
    manifest['git_head']=head
    campaign.update(plan=[],runs=[],tasks_by_job={},engine_manifests={root:manifest},engine_manifests_after={root:manifest})
    for ident in analyzer.ORDER:
        _,repeat,variant=ident.split('.');old_ident=f'task_a.{repeat}.'+('candidate' if variant=='on' else 'baseline')
        job=copy.deepcopy(next(j for j in original['plan'] if j['id']==old_ident))
        run=copy.deepcopy(next(r for r in original['runs'] if r['id']==old_ident))
        worker=json.loads((folder/(old_ident+'.json')).read_text());guard=json.loads((folder/(old_ident+'.guard.json')).read_text())
        command=run['command'][:-len(job['args'])]
        job.update(id=ident,engine_root=root,args=['--expert-batch','--packed-direct-io','--bulk-metadata',
            '--zero-copy-weights','--string-buffer-bytes','65536']+(['--expert-pipeline'] if variant=='on' else []))
        output=folder/(ident+'.json')
        for option,value in (('--engine-root',root),('--label',ident),('--output',str(output)),
                             ('--expected-engine-head',head)):
            command[command.index(option)+1]=value
        command+=job['args'];run.update(id=ident,command=command,guard_file=str(output.with_suffix('.guard.json')))
        task=copy.deepcopy(worker['tasks_snapshot'][0]);task['id']='explanation'
        worker.update(label=ident,engine_root=root,git_head=head,git_head_after=head,
                      command=command[1:]+['--worker'],tasks_snapshot=[task])
        worker['turns'][0]['id']='explanation';worker['turns'][0]['metrics']['cache'].update(bytes=100,capacity=300)
        worker['turns'][0]['metrics']['weights']['expert_pipeline_pairs']=4 if variant=='on' else 0
        worker['backend_options'].update(string_buffer_bytes=65536,expert_pipeline=variant=='on')
        guard['command']=command+['--worker'];save(output,worker);save(output.with_suffix('.guard.json'),guard)
        campaign['plan'].append(job);campaign['runs'].append(run);campaign['tasks_by_job'][ident]=[task]
    save(folder/'campaign.json',campaign)


class HyperstreamAblationTests(unittest.TestCase):
    def test_two_pairs_share_checkout_and_have_descriptive_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);fixture(folder);summary,rows,_=analyzer.analyze_campaign(folder)
            self.assertEqual((summary['pair_count'],summary['worker_count']),(2,4))
            self.assertEqual(summary['aggregate_decode_speedup'],1.5)
            self.assertEqual(len(summary['inputs']),9)
            self.assertEqual(len({r['engine_head'] for r in rows}),1)

    def test_guard_cache_and_unexercised_pipeline_fail(self):
        for mode in ('guard','cache','pipeline'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary);fixture(folder)
                path=folder/('explanation.r0.on.guard.json' if mode=='guard' else 'explanation.r0.on.json')
                value=json.loads(path.read_text())
                if mode=='guard':value['violation']='macos_memory_pressure'
                elif mode=='cache':value['turns'][0]['metrics']['cache']['hits']+=1
                else:value['turns'][0]['metrics']['weights']['expert_pipeline_pairs']=0
                save(path,value)
                with self.assertRaises(ValueError):analyzer.analyze_campaign(folder)

    def test_explicit_public_revision_is_checked_exactly(self):
        public='83daeefde83d1714a2cc2ec1579c3c1c66e98a49'
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);fixture(folder,head=public)
            with self.assertRaises(ValueError):analyzer.analyze_campaign(folder)
            summary,_,_=analyzer.analyze_campaign(folder,expected_head=public)
            self.assertEqual(summary['expected_engine_head'],public)
            with self.assertRaises(ValueError):analyzer.analyze_campaign(folder,expected_head='HEAD')


if __name__=='__main__':unittest.main()
