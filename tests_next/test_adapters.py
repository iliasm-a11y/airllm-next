"""Offline metadata and recurrence contracts; no checkpoint downloads."""
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from airllm.nextgen.adapters import get_adapter, inspect_local_checkpoint, quantization_descriptor, qwen_layer_types
from airllm.nextgen.recurrent import gated_delta_rule, RecurrentState


def qwen_config():
    return dict(model_type='qwen3_5_moe_text', num_hidden_layers=92,
                full_attention_interval=4, linear_num_key_heads=16,
                linear_num_value_heads=128, linear_key_head_dim=128,
                linear_value_head_dim=128, linear_conv_kernel_dim=4,
                mamba_ssm_dtype='float32', num_key_value_heads=4,
                head_dim=256, dtype='bfloat16', mtp_num_hidden_layers=1)


def llama_config():
    return dict(model_type='llama',num_hidden_layers=3,num_attention_heads=8,
                num_key_value_heads=2,hidden_size=128,head_dim=16,
                intermediate_size=256,vocab_size=96,max_position_embeddings=4096)


class AdapterTests(unittest.TestCase):
    def test_qwen_recognition_does_not_claim_execution(self):
        c = qwen_config(); a = get_adapter(c)
        self.assertEqual(qwen_layer_types(c).count('linear_attention'), 69)
        self.assertEqual(qwen_layer_types(c).count('full_attention'), 23)
        with self.assertRaisesRegex(NotImplementedError, 'generation is not'):
            a.require_runtime(c)

    def test_qwen_state_budget_counts_fixed_recurrence(self):
        c = qwen_config(); a = get_adapter(c)
        zero = a.context_bytes(c, 0); context = a.context_bytes(c, 4096)
        self.assertEqual(zero['recurrent_bytes'], 578813952)
        self.assertEqual(context['kv_bytes'], 4096 * 23 * 2 * 4 * 256 * 2)
        self.assertEqual(zero['rollback_snapshot_bytes'], zero['recurrent_bytes'] + zero['convolution_bytes'])
        self.assertGreater(zero['payload_bytes'], 0)

    def test_unknown_and_malformed_schedules_fail(self):
        with self.assertRaises(NotImplementedError): get_adapter({'model_type': 'unimplemented_architecture'})
        c = qwen_config(); c['layer_types'] = ['linear_attention']
        with self.assertRaises(ValueError): qwen_layer_types(c)

    def test_modelopt_is_not_affine_int4(self):
        c = qwen_config(); c['quantization_config'] = {'quant_method': 'modelopt'}
        q = quantization_descriptor(c)
        self.assertEqual(q['format'], 'modelopt_mixed')
        self.assertFalse(q['direct_next_read'])
        with self.assertRaises(NotImplementedError):
            get_adapter({'model_type':'glm_moe_dsa'}).require_runtime(c)

    def test_offline_inspect_does_not_open_or_require_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); (p/'config.json').write_text(json.dumps(qwen_config()))
            (p/'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': 4892365451008.0}, 'weight_map': {'x': 'missing.safetensors'}}))
            r=inspect_local_checkpoint(p)
            self.assertFalse(r['declared_payload_fits_budget'])
            self.assertFalse(r['generation_validated']); self.assertFalse(r['runtime_available'])
            self.assertFalse(r['downloads_performed']); self.assertFalse(r['mtp_execution_available'])
            self.assertEqual(r['declared_tensor_payload_bytes'],4892365451008)

    def test_glm_state_estimate_and_quant_gate(self):
        c=dict(model_type='glm_moe_dsa',num_hidden_layers=3,indexer_types=['full','shared','full'],
               kv_lora_rank=4,qk_rope_head_dim=2,index_head_dim=4)
        a=get_adapter(c); self.assertIs(a.require_runtime(c),a)
        self.assertEqual(a.context_bytes(c,10)['payload_bytes'], (3*6+2*4)*10*2)

    def test_llama_state_estimate_and_architecture_guards(self):
        c=llama_config();adapter=get_adapter(c)
        self.assertIs(adapter.require_runtime(c),adapter)
        self.assertEqual(adapter.context_bytes(c,10)['kv_bytes'],3*2*2*16*10*2)
        for changes in ({'attention_bias':True},{'rope_scaling':{'rope_type':'llama3'}},
                        {'quantization_config':{'quant_method':'fp8','fmt':'e4m3'}}):
            with self.assertRaises(NotImplementedError):adapter.require_runtime(c|changes)

    def test_standalone_inspection_reads_metadata_without_binary_weights(self):
        from airllm.nextgen.lossless import PACK_FORMAT
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text(json.dumps(llama_config()))
            manifest=dict(format=PACK_FORMAT,standalone=True,raw_bytes=100000,
                          stored_bytes=10000,index_bytes=100,assets={})
            (root/'manifest.json').write_text(json.dumps(manifest))
            report=inspect_local_checkpoint(root,storage_budget_bytes=20000)
            self.assertTrue(report['runtime_available'])
            self.assertFalse(report['generation_validated'])
            self.assertEqual(report['declared_tensor_payload_bytes'],100000)
            self.assertFalse(report['declared_payload_fits_budget'])
            storage=report['storage_representation']
            self.assertEqual(storage['kind'],'lossless_standalone')
            self.assertTrue(storage['declared_representation_fits_budget'])
            self.assertFalse(storage['weight_files_present'])
            self.assertFalse(storage['integrity_verified'])
            manifest['stored_bytes']=-1
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):inspect_local_checkpoint(root)

    def test_llama_inspection_rejects_unsupported_runtime_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text(json.dumps(llama_config()|{'attention_bias':True}))
            report=inspect_local_checkpoint(root)
            self.assertFalse(report['runtime_available'])
            self.assertIn('bias-free',report['runtime_rejection_reason'])


class RecurrentTests(unittest.TestCase):
    def values(self):
        r=np.random.default_rng(106)
        q=r.normal(size=(1,7,2,3)).astype(np.float32)
        k=r.normal(size=q.shape).astype(np.float32)
        v=r.normal(size=(1,7,4,5)).astype(np.float32)
        g=-r.uniform(size=(1,7,4)).astype(np.float32)
        b=r.uniform(size=g.shape).astype(np.float32)
        return q,k,v,g,b

    def test_chunking_is_equivalent_to_single_sequence(self):
        q,k,v,g,b=self.values()
        expected,state=gated_delta_rule(q,k,v,g,b,normalize_qk=True)
        pieces=[]; prefix=None
        for start,stop in [(0,2),(2,3),(3,7)]:
            out,prefix=gated_delta_rule(q[:,start:stop],k[:,start:stop],v[:,start:stop],
                                       g[:,start:stop],b[:,start:stop],initial_state=prefix,normalize_qk=True)
            pieces.append(out)
        np.testing.assert_allclose(np.concatenate(pieces,axis=1),expected,rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(prefix,state,rtol=2e-6,atol=2e-6)

    def test_recurrence_matches_affine_matrix_update(self):
        # Independent algebra: S' = (I - beta*k*k^T)*decay*S + beta*k*v^T.
        q,k,v,g,b=self.values(); actual,final=gated_delta_rule(q,k,v,g,b)
        q=np.repeat(q,2,axis=2); k=np.repeat(k,2,axis=2)
        state=np.zeros((1,4,3,5),dtype=np.float64); expected=[]
        for t in range(7):
            kk=k[:,t].astype(np.float64); vv=v[:,t].astype(np.float64)
            a=np.eye(3)-b[:,t,:,None,None]*(kk[...,None]*kk[...,None,:])
            state=(a@state)*np.exp(g[:,t].astype(np.float64))[...,None,None]+b[:,t,:,None,None]*kk[...,None]*vv[...,None,:]
            expected.append(np.einsum('bhk,bhkv->bhv',q[:,t]/np.sqrt(3),state))
        np.testing.assert_allclose(actual,np.stack(expected,axis=1),rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(final,state,rtol=2e-5,atol=2e-5)

    def test_rollback_restores_both_state_banks(self):
        c=RecurrentState(); c.position=3
        c.recurrent[0]=np.ones((2,3),dtype=np.float32);c.convolution[0]=np.ones((2,4),dtype=np.float32)*2
        with self.assertRaises(MemoryError): c.snapshot(max_bytes=1)
        s=c.snapshot(max_bytes=c.nbytes);c.recurrent[0][:]=0;c.convolution[0][:]=0;c.position=9
        c.restore(s)
        self.assertEqual(c.position,3);self.assertTrue((c.recurrent[0]==1).all());self.assertTrue((c.convolution[0]==2).all())
        c.recurrent[0][:]=8;self.assertTrue((s.recurrent[0]==1).all())
        with self.assertRaises(TypeError): c.restore(3)

    def test_shapes_and_empty_sequence(self):
        q,k,v,g,b=self.values()
        with self.assertRaises(ValueError):gated_delta_rule(q,k,v,-g,b)
        out,state=gated_delta_rule(q[:,:0],k[:,:0],v[:,:0],g[:,:0],b[:,:0])
        self.assertEqual(out.shape,(1,0,4,5));self.assertTrue((state==0).all())

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Official recurrence parity requires torch')
    def test_matches_pinned_official_torch_recurrence(self):
        import torch
        path=Path(__file__).resolve().parents[1]/'evidence-ooc/sources/qwen_recurrent_reference.py'
        spec=importlib.util.spec_from_file_location('qwen_pinned_reference',path)
        ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
        q,k,v,g,b=self.values(); q=np.repeat(q,2,axis=2);k=np.repeat(k,2,axis=2)
        state=np.random.default_rng(88).normal(size=(1,4,3,5)).astype(np.float32)
        initial=state.copy()
        for normalize in (False,True):
            actual,final=gated_delta_rule(q,k,v,g,b,initial_state=state,normalize_qk=normalize)
            expected,official_state=ref.torch_recurrent_gated_delta_rule(*[torch.from_numpy(x) for x in (q,k,v,g,b)],
                initial_state=torch.from_numpy(state),output_final_state=True,use_qk_l2norm_in_kernel=normalize)
            np.testing.assert_allclose(actual,expected.numpy(),rtol=2e-5,atol=2e-5)
            np.testing.assert_allclose(final,official_state.numpy(),rtol=2e-5,atol=2e-5)
        np.testing.assert_array_equal(state,initial)


if __name__ == '__main__': unittest.main()
