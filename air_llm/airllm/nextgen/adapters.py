"""Architecture inventory and offline capability checks, without downloading weights.

Recognition is deliberately separate from execution support. Inspecting a Qwen
checkpoint must never dispatch it through GLM attention or an affine INT4 kernel.
"""
from dataclasses import asdict, dataclass
import argparse
import json
import math
import numbers
from pathlib import Path


def validated_default_rope(config):
    """Reject unsupported scaling under either current or legacy spelling."""
    parameters=config.get('rope_parameters')
    if parameters is None:parameters={}
    if not isinstance(parameters,dict):raise ValueError('rope_parameters must be an object')
    declared=[parameters[key] for key in ('rope_type','type') if key in parameters]
    if any(value!='default' for value in declared) or config.get('rope_scaling'):
        raise NotImplementedError('Only default RoPE is implemented; unsupported scaling/aliases cannot be ignored')
    theta=parameters.get('rope_theta',config.get('rope_theta',10000.))
    if not isinstance(theta,numbers.Real) or isinstance(theta,bool) or not math.isfinite(theta) or theta<=0:
        raise ValueError('rope_theta must be finite and positive')
    return dict(parameters)


def validated_llama_rope(config):
    """Normalize only full-head default or linear-scaled Llama RoPE.

    Linear scaling divides positions (equivalently inverse frequencies) by the
    positive factor. It has no sequence-length dependent update or attention
    rescaling. Current/legacy dictionaries may complement each other, but any
    conflicting declaration fails before a checkpoint or device is opened.
    GLM and Qwen continue to use the stricter default-only validator above.
    """
    merged = {}
    allowed = {'rope_type', 'type', 'rope_theta', 'factor', 'original_max_position_embeddings'}
    def positive(value, name):
        try:
            finite = math.isfinite(value)
        except (TypeError, OverflowError):
            finite = False
        if not isinstance(value, numbers.Real) or isinstance(value, bool) or not finite or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if 'rope_theta' in config:
        positive(config['rope_theta'], 'rope_theta')
    for name in ('rope_parameters', 'rope_scaling'):
        values = config.get(name)
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f'{name} must be an object')
        unknown = set(values)-allowed
        if unknown:
            raise NotImplementedError('Unsupported Llama RoPE parameters: '+', '.join(sorted(unknown)))
        for key in ('rope_theta', 'factor'):
            if key in values:
                positive(values[key], key)
        if 'original_max_position_embeddings' in values:
            original = values['original_max_position_embeddings']
            if type(original) is not int or original < 1:
                raise ValueError('original_max_position_embeddings must be a positive integer')
        kinds = [values[key] for key in ('rope_type', 'type') if key in values]
        if len(kinds) == 2 and kinds[0] != kinds[1]:
            raise ValueError('Conflicting Llama RoPE type aliases')
        normalized = {key: value for key, value in values.items() if key != 'type'}
        if kinds:
            normalized['rope_type'] = kinds[0]
        for key, value in normalized.items():
            if key in merged and merged[key] != value:
                raise ValueError(f'Conflicting Llama RoPE declarations for {key}')
            merged[key] = value
    kind = merged.get('rope_type', 'default')
    if kind not in ('default', 'linear'):
        raise NotImplementedError('Next Llama implements default and linear RoPE only')
    theta = merged.get('rope_theta', config.get('rope_theta', 10000.))
    if 'rope_theta' in merged and 'rope_theta' in config and config['rope_theta'] != theta:
        raise ValueError('Conflicting Llama rope_theta declarations')
    positive(theta, 'rope_theta')
    if kind == 'linear' and 'factor' not in merged:
        raise ValueError('Linear Llama RoPE requires an explicit factor')
    factor = merged.get('factor', 1.)
    positive(factor, 'Llama RoPE factor')
    if not math.isfinite(1./factor):
        raise ValueError('Llama RoPE factor has no finite inverse')
    if kind == 'default' and factor != 1:
        raise NotImplementedError('A nonunit Llama RoPE factor requires rope_type=linear')
    original = merged.get('original_max_position_embeddings')
    if original is not None and (type(original) is not int or original < 1):
        raise ValueError('original_max_position_embeddings must be a positive integer')
    result = {'rope_type': kind, 'rope_theta': float(theta)}
    if kind == 'linear':
        result['factor'] = float(factor)
    if original is not None:
        result['original_max_position_embeddings'] = original
    return result


@dataclass(frozen=True)
class ModelAdapter:
    model_type: str
    family: str
    runtime: str | None
    attention: tuple[str, ...]
    recurrent: bool
    mtp_execution: bool = False
    full_checkpoint_validated: bool = False
    metal_validated: bool = False
    experimental_runtime: str | None = None

    def require_runtime(self, config):
        """Fail before creating an MLX device, opening shards, or downloading."""
        if not self.runtime:
            raise NotImplementedError(
                f'{self.model_type} is recognized, but its production runtime '
                'is not enabled. The explicit next-qwen-experimental backend '
                'has miniature floating-checkpoint validation only; full '
                'checkpoint generation is not certified.')
        q = config.get('quantization_config') or {}
        if self.model_type=='llama':
            # Keep inspection aligned with this separate floating Llama path;
            # GLM's FP8 implementation does not enable FP8 for every adapter.
            if q or config.get('quantization'):
                raise NotImplementedError('Next Llama accepts floating checkpoint weights only')
            validated_llama_rope(config)
            if config.get('attention_bias') or config.get('mlp_bias') or config.get('hidden_act','silu')!='silu':
                raise NotImplementedError('Next Llama requires bias-free SiLU layers')
            if config.get('partial_rotary_factor',1.)!=1. or config.get('sliding_window') or config.get('pretraining_tp',1)!=1:
                raise NotImplementedError('Next Llama requires full-head default/linear RoPE, full causal attention and pretraining_tp=1')
        if q and not (q.get('quant_method') == 'fp8' and q.get('fmt', 'e4m3') == 'e4m3'):
            raise NotImplementedError(
                f'Native {self.family} cannot execute quantization '
                f'{q.get("quant_method", "unknown")!r}; a 4-bit label does not '
                'imply compatibility with MLX affine INT4.')
        return self

    def context_bytes(self, config, tokens, batch_size=1, element_bytes=2):
        """Configured state payload, excluding allocator/workspace/rollback overhead.

        GLM's latent K/V and full DSA index keys use the runtime compute dtype.
        Qwen DeltaNet state uses the checkpoint's mamba_ssm_dtype (normally F32).
        This is a structural estimate, not measured resident memory.
        """
        if any(type(x) is not int or x < 0 for x in (tokens, batch_size, element_bytes)):
            raise ValueError('State dimensions must be nonnegative integers')
        if batch_size < 1 or element_bytes < 1:
            raise ValueError('batch_size and element_bytes must be positive')
        layers = int(config['num_hidden_layers'])
        if layers<1:raise ValueError('num_hidden_layers must be positive')
        if self.model_type=='llama':
            heads=config['num_attention_heads'];kv_heads=config.get('num_key_value_heads',heads)
            if any(type(value) is not int or value<1 for value in (heads,kv_heads)):
                raise ValueError('Llama attention heads must be positive integers')
            head_dim=config.get('head_dim',config['hidden_size']//heads)
            if type(head_dim) is not int or head_dim<1 or head_dim%2 or heads%kv_heads:
                raise ValueError('Invalid Llama grouped-query or rotary dimensions')
            kv=batch_size*tokens*layers*2*kv_heads*head_dim*element_bytes
            return {'kv_bytes':kv,'recurrent_bytes':0,'convolution_bytes':0,
                    'payload_bytes':kv,'rollback_snapshot_bytes':0}
        if self.model_type == 'glm_moe_dsa':
            schedule = config.get('indexer_types')
            if schedule is None:
                freq = max(1, int(config.get('index_topk_freq', 1)))
                offset = int(config.get('index_skip_topk_offset', 2))
                schedule = ['full' if max(i-offset+1, 0) % freq == 0 else 'shared' for i in range(layers)]
            if len(schedule) != layers or any(v not in ('full', 'shared') for v in schedule):
                raise ValueError('Invalid GLM indexer schedule')
            dimensions = layers * (int(config['kv_lora_rank']) + int(config['qk_rope_head_dim']))
            dimensions += schedule.count('full') * int(config['index_head_dim'])
            kv = batch_size * tokens * dimensions * element_bytes
            return {'kv_bytes': kv, 'recurrent_bytes': 0, 'convolution_bytes': 0,
                    'payload_bytes': kv, 'rollback_snapshot_bytes': 0}
        schedule = qwen_layer_types(config)
        linear = schedule.count('linear_attention')
        full = schedule.count('full_attention')
        kh = int(config['linear_num_key_heads']); vh = int(config['linear_num_value_heads'])
        kd = int(config['linear_key_head_dim']); vd = int(config['linear_value_head_dim'])
        if min(kh, vh, kd, vd) < 1 or vh % kh:
            raise ValueError('Invalid Qwen recurrent head dimensions')
        state_dtype = config.get('mamba_ssm_dtype', 'float32')
        if state_dtype not in ('float32', 'float16', 'bfloat16'):
            raise NotImplementedError(f'Unknown recurrent state dtype: {state_dtype}')
        state_element = 4 if state_dtype == 'float32' else 2
        recurrent = batch_size * linear * vh * kd * vd * state_element
        # The actual implementation keeps a full kernel-width conv state.
        convolution = batch_size * linear * (2 * kh * kd + vh * vd) * int(config['linear_conv_kernel_dim']) * element_bytes
        kv = batch_size * tokens * full * 2 * int(config['num_key_value_heads']) * int(config['head_dim']) * element_bytes
        return {'kv_bytes': kv, 'recurrent_bytes': recurrent,
                'convolution_bytes': convolution, 'payload_bytes': kv + recurrent + convolution,
                'rollback_snapshot_bytes': recurrent + convolution}


ADAPTERS = {
    'llama':ModelAdapter('llama','Llama full causal GQA','mlx-next-llama',
                         ('full_causal_attention',),False),
    'glm_moe_dsa': ModelAdapter('glm_moe_dsa', 'GLM DSA MoE', 'mlx-next',
                                ('latent_attention', 'dsa_indexer'), False),
    'qwen3_5_moe_text': ModelAdapter('qwen3_5_moe_text', 'Qwen hybrid MoE', None,
                                     ('gated_delta_net', 'gated_attention'), True,
                                     experimental_runtime='next-qwen-experimental'),
}


def get_adapter(config):
    """Look up by architecture identifier, never by an ambiguous marketing name."""
    model_type = config.get('model_type')
    try:
        return ADAPTERS[model_type]
    except (KeyError, TypeError):
        raise NotImplementedError(f'No Next architecture adapter for {model_type!r}') from None


def qwen_layer_types(config):
    layers = config['num_hidden_layers']
    if type(layers) is not int or layers < 1:
        raise ValueError('num_hidden_layers must be positive')
    schedule = config.get('layer_types')
    if schedule is None:
        interval = config.get('full_attention_interval', 4)
        if type(interval) is not int or interval < 1:
            raise ValueError('full_attention_interval must be positive')
        schedule = ['full_attention' if (i + 1) % interval == 0 else 'linear_attention' for i in range(layers)]
    if len(schedule) != layers or any(x not in ('full_attention', 'linear_attention') for x in schedule):
        raise ValueError('Invalid Qwen hybrid layer schedule')
    return list(schedule)


def quantization_descriptor(config):
    q = config.get('quantization_config') or {}
    method = q.get('quant_method')
    if not q:
        return {'format': config.get('dtype', config.get('torch_dtype', 'unspecified')),
                'direct_next_read': True, 'lossless_conversion_certified': False}
    if method == 'fp8' and q.get('fmt', 'e4m3') == 'e4m3':
        return {'format': 'block_fp8_e4m3', 'block': q.get('weight_block_size', [128, 128]),
                'direct_next_read': True, 'activation_quantization_reproduced': False,
                'lossless_conversion_certified': False}
    if method == 'modelopt':
        # NVIDIA ModelOpt files may mix NVFP4 experts, FP8 projections and
        # unquantized sensitive matrices. They are not an affine bit-packed file.
        return {'format': 'modelopt_mixed', 'direct_next_read': False,
                'lossless_conversion_certified': False,
                'reason': 'NVFP4/FP8 layout and scale decoding need separate validated kernels'}
    return {'format': method or 'unknown_quantization', 'direct_next_read': False,
            'lossless_conversion_certified': False, 'reason': 'No validated Next decoder'}


def inspect_local_checkpoint(root, *, tokens=4096, storage_budget_bytes=2_000_000_000_000):
    """Read only JSON metadata. No implicit network, model creation or shard reads.

    metadata.total_size is the tensor payload size declared by the local index,
    not the complete filesystem footprint, nor proof of shard integrity.
    """
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    adapter = get_adapter(config)
    index_path = root / 'model.safetensors.index.json'
    declared = None
    if index_path.exists():
        declared = json.loads(index_path.read_text()).get('metadata', {}).get('total_size')
        # The official Qwen index publishes 4892365451008.0, an exactly
        # representable integral JSON float. Reject fractional/rounded sizes.
        if type(declared) is float and math.isfinite(declared) and declared.is_integer() and abs(declared) <= 2**53:
            declared = int(declared)
        if declared is not None and (type(declared) is not int or declared < 0):
            raise ValueError('Index total_size must be a nonnegative integer')
    if type(storage_budget_bytes) is not int or storage_budget_bytes < 0:
        raise ValueError('storage_budget_bytes must be a nonnegative integer')
    representation={'kind':'original_safetensors','base_checkpoint_required':False,
                    'integrity_verified':False}
    manifest_path=root/'manifest.json'
    if manifest_path.is_file():
        from .lossless import PACK_FORMAT
        manifest=json.loads(manifest_path.read_text())
        if not isinstance(manifest,dict):raise ValueError('Checkpoint manifest must be an object')
        if manifest.get('format')==PACK_FORMAT:
            if manifest.get('standalone') is not True:
                raise ValueError('A partial lossless overlay requires the original checkpoint for inspection')
            for field in ('raw_bytes','stored_bytes','index_bytes'):
                if type(manifest.get(field)) is not int or manifest[field]<0:
                    raise ValueError('Lossless manifest byte counts must be nonnegative integers')
            assets=manifest.get('assets',{})
            if not isinstance(assets,dict) or any(not isinstance(entry,dict) or type(entry.get('bytes')) is not int or entry['bytes']<0 for entry in assets.values()):
                raise ValueError('Invalid lossless asset inventory')
            declared=manifest['raw_bytes']
            representation={'kind':'lossless_standalone','base_checkpoint_required':False,
                'declared_encoded_weight_bytes':manifest['stored_bytes'],
                'declared_index_bytes':manifest['index_bytes'],
                'declared_asset_bytes':sum(entry['bytes'] for entry in assets.values()),
                'weight_files_present':(root/'weights.bin').is_file() and (root/'blocks.idx').is_file(),
                'integrity_verified':False}
            representation['declared_representation_bytes']=(manifest['stored_bytes']+manifest['index_bytes']+
                representation['declared_asset_bytes']+manifest_path.stat().st_size)
            representation['declared_representation_fits_budget']=representation['declared_representation_bytes']<=storage_budget_bytes
    quant = quantization_descriptor(config)
    runtime_rejection=None
    try:adapter.require_runtime(config)
    except (NotImplementedError,ValueError) as error:runtime_rejection=str(error)
    return {'adapter': asdict(adapter), 'quantization': quant,
            'runtime_available': runtime_rejection is None and quant['direct_next_read'],
            'runtime_rejection_reason':runtime_rejection,
            'storage_representation':representation,
            'generation_validated': False,
            'declared_tensor_payload_bytes': declared,
            'storage_budget_bytes': storage_budget_bytes,
            'declared_payload_fits_budget': declared <= storage_budget_bytes if declared is not None else None,
            'context_estimate': adapter.context_bytes(config, tokens),
            'mtp_layers_declared': config.get('mtp_num_hidden_layers', config.get('num_nextn_predict_layers', 0)),
            'mtp_execution_available': adapter.mtp_execution,
            'downloads_performed': False}


def main(argv=None):
    p = argparse.ArgumentParser(description='Offline Next architecture and storage inspection; never inference')
    p.add_argument('model'); p.add_argument('--tokens', type=int, default=4096)
    p.add_argument('--storage-budget-bytes', type=int, default=2_000_000_000_000)
    args = p.parse_args(argv)
    print(json.dumps(inspect_local_checkpoint(args.model, tokens=args.tokens,
                     storage_budget_bytes=args.storage_budget_bytes), indent=2))


if __name__ == '__main__':
    main()
