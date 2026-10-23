"""Aggregation arithmetic can run standalone without importing MLX or Torch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

_file=Path(__file__).resolve().parents[1]/'benchmarks/analyze_native_gguf.py'
_spec=importlib.util.spec_from_file_location('native_gguf_analysis',_file)
analysis=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(analysis)


def run(ids,times,generation=None):
    return {'input_ids':[1],'generated_ids':ids,'token_seconds':times,
        'generation_seconds':sum(times) if generation is None else generation,'wall_seconds':sum(times)+1,
        'logical_bytes':100*len(ids),'cpu_seconds':1,'process_storage_read_bytes':None,
        'options':{'cache_bytes':256},'metrics':{'cache':{'hits':2,'misses':8,'peak_bytes':64}},
        '_guard':{'sampled_peak_rss_bytes':100,'maximum_swap_growth_bytes':4,'minimum_available_bytes':999,'elapsed_seconds':sum(times)+2,
                  'before':{'swap_used_bytes':200}}}


class AnalysisTests(unittest.TestCase):
    def test_rates_pool_tokens_and_seconds_and_missing_storage_is_not_zero(self):
        a=run([2,3],[.5,1.5]);b=run(list(range(8)),[1]+[.5]*7)
        out=analysis.aggregate([a,b])
        self.assertAlmostEqual(out['pooled_generation_tokens_per_second'],10/6.5)
        self.assertAlmostEqual(out['pooled_decode_tokens_per_second'],8/5)
        self.assertAlmostEqual(out['pooled_supervised_process_tokens_per_second'],10/10.5)
        self.assertAlmostEqual(out['median_ttft_seconds'],.75)
        self.assertAlmostEqual(out['pooled_cache_hit_rate'],.2)
        self.assertIsNone(out['process_storage_read_bytes'])
        self.assertEqual(out['process_storage_read_bytes_observed_sum'],0)

    def test_nonfinite_or_missing_supervised_elapsed_cannot_create_a_rate(self):
        a=run([2,3],[.5,1.5]);b=run([2,3],[.5,1.5]);b['_guard']['elapsed_seconds']=float('nan')
        out=analysis.aggregate([a,b])
        self.assertIsNone(out['pooled_supervised_process_tokens_per_second'])
        self.assertEqual(out['supervised_elapsed_measured_runs'],1)
        del b['_guard']['elapsed_seconds']
        self.assertIsNone(analysis.aggregate([a,b])['supervised_process_seconds'])

    def test_paired_speedups_reject_different_ids_or_budgets(self):
        base=run([2,3],[2,2]);fast=run([2,3],[1,1]);wrong=run([2,4],[1,1])
        index={('a',0,'reference'):base,('a',0,'metal'):fast,('b',0,'reference'):base,('b',0,'metal'):wrong}
        out=analysis.paired_speedups(index,['reference','metal'])['metal']
        self.assertEqual(out['eligible_pairs'],1);self.assertEqual(len(out['excluded_pairs']),1)
        self.assertEqual(out['pooled_generation_speedup'],2)
        self.assertEqual(out['pooled_supervised_process_speedup'],1.5)
        fast['options']['cache_bytes']=512
        self.assertEqual(analysis.paired_speedups(index,['reference','metal'])['metal']['eligible_pairs'],0)

    def test_eos_is_included_and_reference_is_not_extrapolated(self):
        self.assertEqual(analysis.eos_prefix([20,248046,198,248045]),[20,248046])
        self.assertEqual(analysis.eos_prefix([1,2,3,4,5]),[1,2,3,4])

    def test_logits_report_full_vector_error_not_only_argmax(self):
        out=analysis.compare_logits([1,3,2],[1,3,2.1],atol=.001,rtol=.001)
        self.assertEqual(out['status'],'failed');self.assertTrue(out['argmax_equal'])
        self.assertAlmostEqual(out['maximum_absolute_error'],.1)
        self.assertEqual(analysis.compare_logits([1,np.nan],[1,2])['status'],'failed')
        self.assertEqual(analysis.compare_logits([1],[1,2])['reason'],'shape_mismatch')

    def test_campaign_with_incomplete_reports_stays_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'paired').mkdir()
            campaign={'tokens_limit':8,'repeats':2,'seed':1,'cases':[{'name':'x','input_ids':[1]}],
                      'variants':{'reference':['reference',2048,None],'metal':['metal',2048,None]},'completed':[]}
            (root/'paired/campaign.json').write_text(json.dumps(campaign))
            out=analysis.analyze(root)
            self.assertEqual(out['status'],'pending');self.assertFalse(out['campaign_complete'])
            self.assertEqual(out['completed_runs'],0);self.assertEqual(len(out['pending_runs']),4)
            self.assertIn('PROVISOIRE',analysis.markdown(out))

    def test_final_eighth_vector_is_not_compared_to_fourth_reference_vector(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);native=root/'native.json';reference=root/'ref.json'
            native.write_text(json.dumps({'input_ids':[1],'generated_ids':list(range(8))}))
            reference.write_text(json.dumps({'input_ids':[1],'generated_ids':list(range(4))}))
            for report in (native,reference):report.with_suffix('.guard.json').write_text(json.dumps({'violation':None,'exit_code':0}))
            np.save(root/'native.npy',np.array([1.,2.]))
            np.savez(root/'ref.npz',logits=np.zeros((4,2)),generated_ids=np.arange(4))
            out=analysis.compare_dump(native,root/'native.npy',reference,root/'ref.npz')
            self.assertEqual(out['status'],'not_compared')
            self.assertEqual(out['reason'],'final_position_exceeds_independent_reference')

    def test_analysis_output_cannot_alias_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);evidence=root/'evidence';evidence.mkdir()
            source=evidence/'keep.json';source.write_text('keep')
            with self.assertRaises(ValueError):analysis.main(['--evidence',str(evidence),'--output',str(source),'--markdown',str(root/'x.md')])
            self.assertEqual(source.read_text(),'keep')

    def test_long_tradeoff_preserves_exact_ids_and_negative_speed_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,n,speed,logical,cpu in [('long64-baseline',64,1.,50,10),('long128-baseline',128,1.,100,20),('long128-decay',128,.95,87,21)]:
                report=root/f'{name}.json'
                report.write_text(json.dumps({'input_ids':[1],'generated_ids':list(range(n)),
                    'generation_tokens_per_second':speed,'decode_tokens_per_second':speed,
                    'logical_bytes':logical,'cpu_seconds':cpu}))
                report.with_suffix('.guard.json').write_text(json.dumps({'violation':None,'exit_code':0}))
            out=analysis.long_comparison(root)
            self.assertEqual(out['status'],'passed');self.assertTrue(out['all_128_ids_equal']);self.assertTrue(out['prefix_64_equal'])
            self.assertAlmostEqual(out['logical_byte_reduction_percent'],13)
            self.assertAlmostEqual(out['generation_throughput_change_percent'],-5)
            self.assertFalse(out['default_change_supported'])


if __name__=='__main__':unittest.main()
