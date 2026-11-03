"""Source/metadata checks only. No compiler, native library, GPU or model call."""
import difflib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tail_even16_build',ROOT/'build_helper.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
prepared=json.loads((ROOT/'prepare-manifest.json').read_text())
base=Path(prepared['original_helper_source'])
variant=ROOT/prepared['variant_helper_source']

def intended_schedule(length, ub=16):
    """Independent policy specification, not compiled C++ execution."""
    full,remainder=divmod(length,ub)
    tail=[] if not remainder else ([remainder-remainder%8,remainder%8] if remainder>8 and remainder%2 else [remainder])
    return [ub]*full+tail

def parity_signature(sizes):
    """Merged token intervals eligible for nrc2 by column parity only."""
    result=[];offset=0
    for count in sizes:
        eligible=2 if count%2==0 else 1
        if result and result[-1][2]==eligible:result[-1]=(result[-1][0],offset+count,eligible)
        else:result.append((offset,offset+count,eligible))
        offset+=count
    return result

def cpu8_signature(length):
    remainder=length%8
    scalar=remainder if remainder%2 else 0
    prefix=length-scalar
    return ([(0,prefix,2)] if prefix else [])+([(prefix,length,1)] if scalar else [])

class StructuralTests(unittest.TestCase):
    def test_original_dependencies_and_comparators_are_unchanged(self):
        self.assertEqual(b.sha(base),prepared['original_helper_source_sha256'])
        self.assertEqual(b.sha(prepared['original_helper_binary']),prepared['original_helper_binary_sha256'])
        for name,digest in prepared['unchanged_copied_dependencies'].items():self.assertEqual(b.sha(ROOT/name),digest)
        for name,digest in prepared['existing_comparator_hashes'].items():self.assertEqual(b.sha(base.parent/name),digest)
    def test_patch_is_exact_isolated_helper_delta(self):
        expected=''.join(difflib.unified_diff(base.read_text().splitlines(True),variant.read_text().splitlines(True),
            fromfile='a/research/glm53_batched_prefill/native_prefill.cpp',tofile='b/research/glm53_batched_prefill/native_prefill.cpp'))
        self.assertEqual((ROOT/'patches/native-prefill-tail-even16.patch').read_text(),expected)
        text=variant.read_text()
        self.assertIn('ub != 8 && ub != 16 && ub != 32 && ub != 64 && ub != 128',text)
        self.assertIn('return remaining < ub && take > 8 && take % 2 != 0 ? take - take % 8 : take;',text)
        self.assertEqual(text.count('const int take=prefill_take(ub,end-offset);'),1)
        self.assertEqual(text.count('prefill_chunk_policy_qualified'),1)
    def test_intended_schedule_preserves_cpu8_parity_over_all_profiles(self):
        self.assertEqual(intended_schedule(521),[16]*32+[8,1])
        self.assertEqual(intended_schedule(523),[16]*32+[8,3])
        self.assertEqual(intended_schedule(515),[16]*32+[3])
        self.assertEqual(intended_schedule(542),[16]*33+[14])
        self.assertEqual(intended_schedule(208),[16]*13)
        self.assertEqual(intended_schedule(521-208),[16]*19+[8,1])
        self.assertNotEqual(parity_signature([16]*32+[10,1]),cpu8_signature(523))
        for ub in (8,16,32,64,128):
            for total in range(1,8193):
                sizes=intended_schedule(total,ub)
                self.assertEqual(sum(sizes),total)
                self.assertTrue(all(0<value<=ub for value in sizes))
                self.assertEqual(sizes[:total//ub],[ub]*(total//ub))
                self.assertEqual(parity_signature(sizes),cpu8_signature(total))
    def test_logits_snapshot_and_checkpoint_bodies_are_unchanged(self):
        old,new=base.read_text(),variant.read_text()
        for start,end in [('static json usage()','struct observer'),('const auto budget=','auto decode='),
            ('auto batch=llama_batch_get_one','o.batches<<json'),('if(rc)throw','if(mode=="restore")')]:
            left=old[old.index(start):old.index(end,old.index(start))]
            right=new[new.index(start):new.index(end,new.index(start))]
            self.assertEqual(left,right)
        self.assertEqual(new.count('if(offset+take==int(ids[request].size()))output_flags.back()=1;'),1)
        self.assertEqual(new.count('snapshot.capture(ctx,key,208,scratch,budget);'),1)
    def test_build_plan_does_not_start_any_process(self):
        reconstruction=json.loads(Path(prepared['native_source_reconstruction_receipt']).read_text())
        baseline=json.loads(Path(prepared['original_build_receipt']).read_text())
        command=baseline['command'];libraries=command[command.index('-L')+1]
        with mock.patch.object(b.subprocess,'run',side_effect=AssertionError('no process in plan')) as spawned:
            record=b.plan(reconstruction['source'],libraries,ROOT/'build/native-prefill-cpu8-tail-compatible')
        spawned.assert_not_called()
        self.assertEqual(record['status'],'planned_not_built')
        self.assertFalse(record['build_attempted']);self.assertFalse(record['build_performed'])
        self.assertFalse(record['model_loaded']);self.assertFalse(record['full_model_qualified'])
        self.assertEqual(set(record['header_paths']),set(b.HEADER_NAMES))
        for relative,path in record['header_paths'].items():self.assertEqual(record['inputs_sha256'][path],record['header_sha256'][relative])
        self.assertEqual(record['original_helper_binary_sha256'],prepared['original_helper_binary_sha256'])

if __name__=='__main__':unittest.main()
