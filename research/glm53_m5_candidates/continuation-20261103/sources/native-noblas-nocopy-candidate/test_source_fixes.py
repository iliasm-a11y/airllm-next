"""Prepared negatives for the review fixes. NOT executed during CPU v4."""
from pathlib import Path
import copy
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch
import gpu_contract as g


def complete_document():
    return {'template':'synthetic schema test, not model evidence',
        'agents':[{'agent_id':i,'token_ids':[0]*208+[i]} for i in range(1,601)]}


class FullInputTests(unittest.TestCase):
    def test_complete_schema_requires_all_ids(self):
        self.assertEqual(len(g.validate_full_prompts(complete_document(),g.FROZEN_PROMPTS_SHA256)),600)

    def test_two_agent_input_refused(self):
        d=complete_document();d['agents']=[d['agents'][0],d['agents'][-1]]
        with self.assertRaises(g.GpuContractError):g.validate_full_prompts(d,g.FROZEN_PROMPTS_SHA256)

    def test_duplicate_integer_id_refused(self):
        d=complete_document();d['agents'][-1]['agent_id']=1
        with self.assertRaises(g.GpuContractError):g.validate_full_prompts(d,g.FROZEN_PROMPTS_SHA256)

    def test_bool_id_is_not_integer_id(self):
        d=complete_document();d['agents'][0]['agent_id']=True
        with self.assertRaises(g.GpuContractError):g.validate_full_prompts(d,g.FROZEN_PROMPTS_SHA256)

    def test_substituted_json_refused(self):
        with self.assertRaises(g.GpuContractError):g.validate_full_prompts(complete_document(),'0'*64)

    def test_short_or_noninteger_full_tokens_refused(self):
        for tokens in ([0]*128,[0]*208+[1.0]):
            d=complete_document();d['agents'][300]['token_ids']=tokens
            with self.subTest(tokens=len(tokens)),self.assertRaises(g.GpuContractError):g.validate_full_prompts(d,g.FROZEN_PROMPTS_SHA256)

    def test_no_invented_complete_field(self):
        d=complete_document();self.assertNotIn('complete',d)
        g.validate_full_prompts(d,g.FROZEN_PROMPTS_SHA256)

    def test_foreign_contract_cache_refused_before_build(self):
        here=Path(__file__).resolve().parent
        foreign=types.ModuleType('gpu_contract');foreign.__file__=str(here.parent/'foreign/gpu_contract.py')
        spec=importlib.util.spec_from_file_location('unexecuted_probe_builder_test',here/'build_driver.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'gpu_contract':foreign}),patch('subprocess.run') as run:
            with self.assertRaises(ImportError):spec.loader.exec_module(module)
            run.assert_not_called()


if __name__=='__main__':unittest.main()
