#!/usr/bin/env python3
"""Verify the real GLM-5.3 config/index against a native meta model, no weights."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='evidence/glm-5.3.config.json')
    p.add_argument('--index', default='evidence/glm-5.3.index.json')
    p.add_argument('--output', default='evidence/glm-5.3-layout-audit.json')
    args = p.parse_args()
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    raw = Path(args.config).read_bytes()
    idxraw = Path(args.index).read_bytes()
    config_dict = json.loads(raw)
    index = json.loads(idxraw)
    keys = set(index['weight_map'])
    missing = []
    mappings = []
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 'config.json').write_bytes(raw)
        cfg = AutoConfig.from_pretrained(d, local_files_only=True, trust_remote_code=False)
        cfg._experts_implementation = 'eager'
        with torch.device('meta'), init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False,
                                                     attn_implementation='sdpa', dtype=torch.bfloat16)
        count = sum(p.numel() for p in model.parameters())
        for name, parameter in model.state_dict(keep_vars=True).items():
            if name.endswith('.experts.gate_up_proj') or name.endswith('.experts.down_proj'):
                prefix, attr = name.rsplit('.', 1)
                targets = ('gate_proj', 'up_proj') if attr == 'gate_up_proj' else ('down_proj',)
                expected = [f'{prefix}.{e}.{target}.weight' for e in range(cfg.n_routed_experts)
                            for target in targets]
                missing.extend(k for k in expected if k not in keys)
                mappings.append({'native': name, 'checkpoint_tensors': len(expected),
                                 'native_shape': list(parameter.shape)})
            elif name not in keys:
                missing.append(name)
        expert_keys = [k for k in keys if '.mlp.experts.' in k]
        quant = config_dict['quantization_config']
        fp8_experts = [k for k in expert_keys if k.endswith('.weight')]
        scales_missing = [k + '_scale_inv' for k in fp8_experts if k + '_scale_inv' not in keys]
    report = {'config_url': 'https://huggingface.co/zai-org/GLM-5.3/resolve/main/config.json',
              'index_url': 'https://huggingface.co/zai-org/GLM-5.3/resolve/main/model.safetensors.index.json',
              'retrieved_utc_date': '2026-10-22', 'config_sha256': sha256(raw).hexdigest(),
              'index_sha256': sha256(idxraw).hexdigest(), 'transformers': transformers.__version__,
              'architecture': cfg.architectures, 'native_meta_model_parameters': count,
              'native_meta_model_created': True, 'layers': cfg.num_hidden_layers,
              'experts_per_sparse_layer': cfg.n_routed_experts,
              'selected_experts_per_token': cfg.num_experts_per_tok,
              'checkpoint_bytes': index['metadata']['total_size'],
              'checkpoint_tensor_entries': len(keys),
              'checkpoint_files': len(set(index['weight_map'].values())),
              'fp8_expert_weights_with_scale': len(fp8_experts)-len(scales_missing),
              'missing_model_keys': missing, 'missing_expert_scale_keys': scales_missing,
              'packed_to_individual_mappings': mappings,
              'quantization': {k: quant.get(k) for k in ('quant_method','weight_block_size','fmt','activation_scheme')},
              'full_weights_downloaded': False, 'full_model_inference_tested': False,
              'm5_mps_or_mlx_tested': False,
              'conclusion': 'Native config/meta skeleton and tensor-name layout verified; no full-weight or hardware validation.'}
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'packed_to_individual_mappings'}, indent=2))
    if missing or scales_missing:
        raise SystemExit('Official layout has unresolved keys; support is incomplete')


if __name__ == '__main__':
    main()
