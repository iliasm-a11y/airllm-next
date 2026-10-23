"""Read-only admission planning for local streamed checkpoints.

This is a metadata/resource gate, never a generation certificate. It does not
read tensor payloads, download weights, create cache files, or infer SSD speed.
The configured budgets are reservations; allocator transients and OS pressure
still require runtime enforcement and measurements on the actual machine.
"""
import argparse
from dataclasses import asdict
import json
import math
import platform
from pathlib import Path
import shutil

from .adapters import get_adapter, inspect_local_checkpoint, qwen_layer_types
from .config import NextConfig, GiB


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _local_files(root, inspection):
    """Inventory declared files by stat only; presence is not integrity."""
    if inspection['storage_representation']['kind'] == 'lossless_standalone':
        manifest = json.loads((root / 'manifest.json').read_text())
        names = ['weights.bin', 'blocks.idx', *manifest.get('assets', {})]
    else:
        index = root / 'model.safetensors.index.json'
        if index.is_file():
            data = json.loads(index.read_text())
            mapping = data.get('weight_map')
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError('Checkpoint index needs a nonempty weight_map')
            if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                   for k, v in mapping.items()):
                raise ValueError('Invalid checkpoint weight_map entry')
            names = sorted(set(mapping.values()))
        else:
            names = sorted(path.name for path in root.glob('*.safetensors')) or ['model.safetensors']
    missing, present, empty = [], [], []
    for name in names:
        path = root / name
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or '..' in Path(name).parts or not path.resolve().is_relative_to(root)):
            raise ValueError('Checkpoint file escapes model directory')
        if not path.is_file():
            missing.append(name)
        else:
            size = path.stat().st_size
            present.append({'file': name, 'bytes': size})
            if size == 0:
                empty.append(name)
    return {'required_files': len(names), 'present_files': present,
            'missing_files': missing, 'empty_files': empty,
            'present_file_bytes': sum(item['bytes'] for item in present),
            'all_declared_files_present': not missing and not empty,
            'tensor_headers_checked': False, 'payload_integrity_verified': False}


def preflight_checkpoint(root, config=None, *, tokens=4096, speculative_depth=0,
                         batch_size=1, concurrent_inferences=1,
                         logical_agents=1, available_memory_bytes=None,
                         storage_free_bytes=None, storage_reserve_bytes=GiB,
                         allow_experimental=False):
    """Return explicit blockers before loading a model or starting a download.

    ``tokens`` is the retained prefix length. Speculative proposals are added
    to that length for planning. Logical agents do not multiply weight budgets;
    concurrent inference reservations do. A single engine currently serializes
    execution, so concurrency > 1 is a capacity scenario, not a capability claim.
    ``available_memory_bytes``/``storage_free_bytes`` can describe a different
    target, but an overridden resource value is labelled as supplied, not seen.
    """
    root = Path(root).resolve()
    cfg = config if config is not None else NextConfig()
    if not isinstance(cfg, NextConfig):
        raise TypeError('config must be NextConfig')
    for name, value, minimum in (
            ('tokens', tokens, 0), ('speculative_depth', speculative_depth, 0),
            ('batch_size', batch_size, 1), ('concurrent_inferences', concurrent_inferences, 1),
            ('logical_agents', logical_agents, 1), ('storage_reserve_bytes', storage_reserve_bytes, 0)):
        _integer(name, value, minimum)
    if type(allow_experimental) is not bool:
        raise ValueError('allow_experimental must be boolean')
    if logical_agents < concurrent_inferences:
        raise ValueError('logical_agents cannot be smaller than concurrent_inferences')
    memory_source = 'supplied scenario' if available_memory_bytes is not None else 'psutil available memory on execution host'
    if available_memory_bytes is None:
        import psutil
        available_memory_bytes = int(psutil.virtual_memory().available)
    _integer('available_memory_bytes', available_memory_bytes)
    storage_source = 'supplied scenario' if storage_free_bytes is not None else 'free bytes on local checkpoint filesystem'
    if storage_free_bytes is None:
        storage_free_bytes = shutil.disk_usage(root).free
    _integer('storage_free_bytes', storage_free_bytes)
    model = json.loads((root / 'config.json').read_text())
    inspection = inspect_local_checkpoint(root, tokens=tokens,
                                          storage_budget_bytes=storage_free_bytes)
    adapter = get_adapter(model)
    element_bytes = 4 if cfg.dtype == 'float32' else 2
    state = adapter.context_bytes(model, tokens + speculative_depth,
                                  batch_size=batch_size, element_bytes=element_bytes)
    prior = adapter.context_bytes(model, tokens, batch_size=batch_size,
                                  element_bytes=element_bytes)
    # Experimental Qwen forward snapshots all existing banks for exception
    # rollback, including full-attention KV. Count that entire reservation.
    snapshot = prior['payload_bytes'] if adapter.recurrent and tokens > 0 else 0
    capacity_tokens = tokens + speculative_depth
    allocated_state = state['payload_bytes']
    growth_overlap = 0
    replacement_overlap = 0
    if adapter.recurrent:
        schedule = qwen_layer_types(model)
        full, linear = schedule.count('full_attention'), schedule.count('linear_attention')
        replacement_overlap = max(state['kv_bytes'] // full if full else 0,
                                  state['recurrent_bytes'] // linear if linear else 0,
                                  state['convolution_bytes'] // linear if linear else 0)
    if not adapter.recurrent and capacity_tokens:
        limit = min(cfg.max_context, model.get('max_position_embeddings', cfg.max_context))
        capacity_tokens = max(capacity_tokens, min(limit, math.ceil(capacity_tokens / 128) * 128))
        allocated_state = adapter.context_bytes(model, capacity_tokens,
                batch_size=batch_size, element_bytes=element_bytes)['payload_bytes']
        previous_capacity = 128 * ((capacity_tokens - 1) // 128)
        if adapter.model_type == 'llama':
            heads = model.get('num_key_value_heads', model['num_attention_heads'])
            width = 2 * heads * model.get('head_dim', model['hidden_size'] // model['num_attention_heads'])
        else:
            # At most one growing layer's old bank remains while all other
            # banks can conservatively be counted at their final capacity.
            width = model['kv_lora_rank'] + model['qk_rope_head_dim'] + model['index_head_dim']
        growth_overlap = previous_capacity * width * element_bytes * batch_size
    required_state = allocated_state + growth_overlap + snapshot + replacement_overlap
    per_inference = cfg.io_bytes + cfg.kv_bytes + cfg.workspace_bytes
    requested = cfg.cache_bytes + concurrent_inferences * per_inference + cfg.reserve_bytes
    inventory = _local_files(root, inspection)
    blockers = []
    runtime = inspection['runtime_available']
    if allow_experimental and adapter.experimental_runtime and not model.get('quantization_config') and not model.get('quantization'):
        runtime = True
    if not runtime:
        blockers.append(inspection['runtime_rejection_reason'] or 'No validated weight decoder')
    if batch_size != 1:
        blockers.append('Next runtimes support unpadded batch_size=1 only')
    if speculative_depth and adapter.recurrent:
        blockers.append('Qwen recurrent speculative verification is not enabled')
    if not inventory['all_declared_files_present']:
        blockers.append('Declared weight files are missing or empty')
    if tokens + speculative_depth > cfg.max_context:
        blockers.append('Requested prefix and proposals exceed configured max_context')
    declared_max = model.get('max_position_embeddings')
    if declared_max is not None:
        _integer('max_position_embeddings', declared_max, 1)
        if tokens + speculative_depth > declared_max:
            blockers.append('Requested prefix and proposals exceed model max_position_embeddings')
    if required_state > cfg.kv_bytes:
        blockers.append('State plus retained rollback reservation exceeds kv_bytes')
    if requested > available_memory_bytes:
        blockers.append('Configured inference budgets plus reserve exceed available memory')
    representation = inspection['storage_representation']
    declared = representation.get('declared_representation_bytes', inspection['declared_tensor_payload_bytes'])
    fresh_fit = None if declared is None else declared + storage_reserve_bytes <= storage_free_bytes
    if inventory['all_declared_files_present']:
        download = 'not_needed_for_declared_files'
    elif declared is None:
        download = 'blocked_unknown_payload_size'
    elif not fresh_fit:
        download = 'blocked_fresh_payload_exceeds_free_space_with_reserve'
    else:
        download = 'metadata_only_prerequisites_unverified'
    return {'schema_version': 1, 'kind': 'metadata_resource_preflight',
            'preflight_passed': not blockers, 'blockers': blockers,
            'generation_validated': False, 'full_checkpoint_validated': False,
            'downloads_performed': False, 'writes_performed': False,
            'host': {'platform': platform.platform(), 'machine': platform.machine()},
            'model_type': adapter.model_type, 'inspection': inspection,
            'experimental_requested': allow_experimental,
            'files': inventory,
            'state': state | {'prefix_tokens': tokens, 'proposed_tokens': speculative_depth,
                              'allocated_capacity_tokens': capacity_tokens,
                              'allocated_state_bytes': allocated_state,
                              'kv_growth_overlap_reservation_bytes': growth_overlap,
                              'bank_replacement_reservation_bytes': replacement_overlap,
                              'retained_transaction_snapshot_bytes': snapshot,
                              'required_state_reservation_bytes': required_state,
                              'kv_budget_bytes': cfg.kv_bytes},
            'memory': {'available_bytes': available_memory_bytes, 'source': memory_source,
                       'configured_total_with_reserve_bytes': requested,
                       'logical_agents': logical_agents, 'concurrent_inferences': concurrent_inferences,
                       'shared_weight_cache_bytes': cfg.cache_bytes,
                       'budgets': asdict(cfg), 'is_complete_peak_bound': False},
            'storage': {'free_bytes': storage_free_bytes, 'source': storage_source,
                        'reserve_bytes': storage_reserve_bytes,
                        'fresh_declared_payload_fits_with_reserve': fresh_fit,
                        'download_gate': download, 'ssd_throughput_measured': False},
            'unknowns': ['tensor content and header integrity', 'checkpoint numerical fidelity',
                         'kernel activation quantization semantics', 'allocator/copy transients',
                         'attention and logits workspace peaks', 'tokenizer/host object memory',
                         'runtime memory pressure and swap', 'sustained SSD throughput',
                         'thermal throttling', 'metadata/tokenizer storage overhead for a fresh download']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--tokens', type=int, default=4096)
    parser.add_argument('--speculative-depth', type=int, default=0)
    parser.add_argument('--logical-agents', type=int, default=1)
    parser.add_argument('--allow-experimental', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(preflight_checkpoint(args.model, tokens=args.tokens,
          speculative_depth=args.speculative_depth, logical_agents=args.logical_agents,
          allow_experimental=args.allow_experimental), indent=2))


if __name__ == '__main__':
    main()
