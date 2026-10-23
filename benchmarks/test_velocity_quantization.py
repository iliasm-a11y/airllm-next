"""Recipe/identity checks only: no corpus inference or weight conversion."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC=importlib.util.spec_from_file_location('benchmark_velocity_quantization',
    Path(__file__).with_name('benchmark_velocity_quantization.py'))
BENCHMARK=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(BENCHMARK)


def checkpoint():
    prefix='model.layers.0.block_sparse_moe.'
    return SimpleNamespace(tensors={prefix+'input_linear.weight':{'shape':[2,128,64]},
        prefix+'output_linear.weight':{'shape':[2,64,64]},
        prefix+'router.layer.weight':{'shape':[2,64]},
        'model.layers.0.self_attn.q_proj.weight':{'shape':[64,64]},
        'lm_head.weight':{'shape':[128,64]},
        'model.embed_tokens.weight':{'shape':[128,64]}})


def test_q8_expert_recipe_changes_every_expert_bit_width_and_preserves_scope():
    cp=checkpoint()
    expert8=BENCHMARK.expected_entries(cp,'expert8')
    mixed8=BENCHMARK.expected_entries(cp,'mixed8')
    assert len(expert8)==4 and len(mixed8)==6
    assert all(row[3]==8 for row in mixed8.values())
    assert all('router' not in ident and 'embed_tokens' not in ident for ident in mixed8)
    assert 'lm_head.weight' in mixed8 and 'lm_head.weight' not in expert8
    assert BENCHMARK.payload_bytes(expert8,64)==24576*9//8


def test_group32_q4_increases_metadata_without_changing_selected_matrices():
    cp=checkpoint()
    q4=BENCHMARK.expected_entries(cp,'expert4g32')
    q8=BENCHMARK.expected_entries(cp,'expert8')
    assert set(q4)==set(q8)
    assert all(row[3]==4 for row in q4.values())
    assert BENCHMARK.payload_bytes(q4,32)==24576*3//4
    assert BENCHMARK.payload_bytes(q4,32)>BENCHMARK.payload_bytes(q4,64)
    bad={'x':('x',None,(2,33),4)}
    with pytest.raises(ValueError,match='width'):BENCHMARK.payload_bytes(bad,32)


def test_new_variants_keep_identical_original_quality_budgets(tmp_path):
    args=SimpleNamespace(device='cpu',cache_mib=256,pack_root=tmp_path)
    configs=[BENCHMARK.configuration(args,tmp_path/'metadata',variant)
             for variant in ('original',)+tuple(BENCHMARK.RECIPES)]
    original=configs[0].as_dict();original.pop('quantized_dir')
    for cfg in configs:
        actual=cfg.as_dict();actual.pop('quantized_dir')
        assert actual==original
        assert cfg.dtype=='float32' and cfg.zero_copy_weights is False
    assert configs[0].quantized_dir is None
    assert Path(configs[1].quantized_dir).name=='quantized-granite-expert8'


def test_historical_comparison_requires_identical_windows_and_labels():
    manifest={'segments':[{'token_ids':[1,2]}]}
    summary={'original':{'variant_mean_nll':2.5}}
    prior={'corpus':copy.deepcopy(manifest),'summary':copy.deepcopy(summary)}
    result=BENCHMARK.paired_prior(prior,manifest,summary,512)
    assert result['same_512_label_protocol'] and result['current_reference_matches_prior_nll']
    subset=BENCHMARK.paired_prior(prior,manifest,summary,128)
    assert subset['same_512_label_protocol'] is False
    assert subset['current_reference_matches_prior_nll'] is None
    edited=copy.deepcopy(manifest);edited['segments'][0]['token_ids'][0]=7
    with pytest.raises(ValueError,match='different corpus'):BENCHMARK.paired_prior(prior,edited,summary,512)


def test_sidecar_audit_rejects_incomplete_precisions_and_payloads(tmp_path):
    cp=checkpoint();entries=BENCHMARK.expected_entries(cp,'expert8')
    path=tmp_path/'manifest.json';path.write_text('{}')
    recorded=[]
    def verify(**kwargs):recorded.append(kwargs)
    sidecar=SimpleNamespace(group_size=64,bits=8,path=path,root=tmp_path,
        entries={ident:{'shape':shape,'bits':bits,'nbytes':shape[0]*shape[1]*9//8}
                 for ident,(_,_,shape,bits) in entries.items()},
        verify=verify,stats={'integrity_bytes':123})
    result=BENCHMARK.validate_sidecar(cp,sidecar,'expert8')
    assert result['matrices']==4 and result['payload_bytes']==27648
    assert recorded==[{'require_digest':True}]
    first=next(iter(sidecar.entries));sidecar.entries[first]['bits']=4
    with pytest.raises(ValueError,match='precision'):BENCHMARK.validate_sidecar(cp,sidecar,'expert8')
    sidecar.entries[first]['bits']=8;sidecar.entries[first]['nbytes']+=4
    with pytest.raises(ValueError,match='payload'):BENCHMARK.validate_sidecar(cp,sidecar,'expert8')
    sidecar.entries.pop(first)
    with pytest.raises(ValueError,match='Incomplete'):BENCHMARK.validate_sidecar(cp,sidecar,'expert8')
