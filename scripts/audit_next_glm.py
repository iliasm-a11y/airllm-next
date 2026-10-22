#!/usr/bin/env python3
"""Structural native-GLM audit against actual official config/index and HF meta model.

Does not fetch or allocate any model weights. Shapes inferred from the native
Transformers architecture are not checkpoint-header validation of 755 GB.
"""
import hashlib,json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'air_llm'))
import torch
from transformers import GlmMoeDsaConfig,GlmMoeDsaForCausalLM
from airllm.nextgen.runtime import AirLLMNextModel,normalize_config

root=Path(__file__).resolve().parents[1];src=root/'evidence-next/sources'
c=json.loads((src/'config-glm.json').read_text());idx=json.loads((src/'glm-model-index.json').read_text())['weight_map']
with torch.device('meta'):meta=GlmMoeDsaForCausalLM(GlmMoeDsaConfig(**c))
shapes={k:tuple(v.shape) for k,v in meta.state_dict().items()}
expanded={}
for key,shape in shapes.items():
    if '.experts.' in key and len(shape)==3:
        prefix=key.rsplit('.',1)[0];kind=key.rsplit('.',1)[1]
        for e in range(shape[0]):
            if kind=='gate_up_proj':
                expanded[f'{prefix}.{e}.gate_proj.weight']={'shape':(shape[1]//2,shape[2])}
                expanded[f'{prefix}.{e}.up_proj.weight']={'shape':(shape[1]//2,shape[2])}
            elif kind=='down_proj':expanded[f'{prefix}.{e}.down_proj.weight']={'shape':shape[1:]}
    else:expanded[key]={'shape':shape}
missing=[k for k in expanded if k not in idx]
if missing:raise AssertionError(f'Missing official keys: {missing[:8]}')
probe=object.__new__(AirLLMNextModel);probe.config=normalize_config(c);probe.checkpoint=SimpleNamespace(tensors=expanded)
probe._validate_layout()
report={'official_architecture':c['architectures'],'metadata_keys_matched':len(expanded),'native_parameter_count':sum(p.numel() for p in meta.parameters()),
 'config_sha256':hashlib.sha256((src/'config-glm.json').read_bytes()).hexdigest(),'index_sha256':hashlib.sha256((src/'glm-model-index.json').read_bytes()).hexdigest(),
 'source_revision':json.loads((src/'glm-model-api.json').read_text()).get('sha'),'missing_keys':missing,'native_layout_check_passed':True,
 'weights_downloaded':False,'full_checkpoint_headers_validated':False,'M5_inference_validated':False,
 'shape_source':'Transformers meta architecture; actual checkpoint tensor headers remain to validate on local files',
 'MTP':'one prediction layer advertised, weights not executed by this engine'}
(root/'evidence-next/glm-native-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
