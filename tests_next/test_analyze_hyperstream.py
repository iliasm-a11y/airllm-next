"""Synthetic evidence only; executable with unittest without engine imports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import analyze_hyperstream as analyzer
from test_analyze_research51 import fixture,save


def hyperstream_fixture(folder):
    campaign=fixture(folder)
    for field in ('plan','runs'):
        campaign[field]=[job for job in campaign[field] if not job['id'].startswith('task_c.')]
    campaign['tasks_by_job']={k:v for k,v in campaign['tasks_by_job'].items() if not k.startswith('task_c.')}
    for job,run in zip(campaign['plan'],campaign['runs']):
        candidate=job['id'].endswith('candidate');old=job['args']
        command=run['command'][:-len(old)] if old else run['command'][:]
        job['args']=['--expert-batch','--packed-direct-io','--bulk-metadata','--zero-copy-weights']
        if candidate:job['args']+=['--expert-pipeline','--string-buffer-bytes','65536']
        command+=job['args'];run['command']=command
        path=folder/(job['id']+'.json');worker=json.loads(path.read_text())
        worker['command']=command[1:]+['--worker'];worker['options']['zero_copy_weights']=True
        worker['tokenizer_io']={'read_calls':5 if candidate else 100,'logical_bytes':1000}
        worker['backend_options']=dict(kernel='metal',expert_batch=True,packed_direct_io=True,bulk_metadata=True)
        if candidate:worker['backend_options'].update(expert_pipeline=True,string_buffer_bytes=65536)
        save(path,worker)
        path=path.with_suffix('.guard.json');guard=json.loads(path.read_text());guard['command']=command+['--worker'];save(path,guard)
    save(folder/'campaign.json',campaign)
    return campaign


class HyperstreamAnalyzerTests(unittest.TestCase):
    def test_four_pairs_have_distinct_stream_forward_and_process_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);hyperstream_fixture(folder)
            summary,rows,raw=analyzer.analyze_campaign(folder)
            self.assertEqual((summary['pair_count'],summary['worker_count'],summary['prompt_count']),(4,8,2))
            self.assertEqual(summary['aggregate_decode_speedup'],1.5)
            self.assertIsNone(summary['aggregate']['candidate']['process_storage_read_bytes'])
            self.assertEqual(summary['aggregate']['candidate']['tokenizer_read_calls']['median'],5)
            self.assertEqual(len(summary['inputs']),17)
            analyzer.shared.write_outputs(folder/'analysis',summary,rows,raw)
            self.assertIn('FIXTURES SYNTHÉTIQUES',(folder/'analysis/responses.md').read_text())

    def test_incomplete_or_wrong_pair_count_is_rejected(self):
        for mode in ('stopped','missing'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary);campaign=hyperstream_fixture(folder)
                if mode=='stopped':campaign['status']='stopped'
                else:campaign['plan'].pop();campaign['runs'].pop()
                save(folder/'campaign.json',campaign)
                with self.assertRaises(ValueError):analyzer.analyze_campaign(folder)

    def test_drift_guard_violation_and_uncontrolled_options_are_rejected(self):
        for mode in ('logits','routes','text','guard','options','treatment'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary);hyperstream_fixture(folder)
                path=folder/('task_a.r0.candidate.guard.json' if mode=='guard' else 'task_a.r0.candidate.json')
                value=json.loads(path.read_text())
                if mode in ('logits','routes'):value['turns'][0]['step_checks'][0][mode+'_sha256']='2'*64
                elif mode=='text':value['turns'][0]['text']='altered fixture'
                elif mode=='guard':value['violation']='macos_memory_pressure'
                elif mode=='options':value['options']['dtype']='float16'
                else:value['backend_options']['expert_pipeline']=False
                save(path,value)
                with self.assertRaises(ValueError):analyzer.analyze_campaign(folder)


if __name__=='__main__':unittest.main()
