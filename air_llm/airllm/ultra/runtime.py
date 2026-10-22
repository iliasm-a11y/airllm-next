"""AirLLM's real Transformers model with direct, bounded out-of-core weights.

Opt-in backend: legacy inference/training stays intact. No gc.collect(), allocator
trim or empty_cache() in the layer/token hot path. Python model skeleton and
checkpoint mappings survive generation calls. This wrapper is single-request;
calls on one instance are serialized because hooks mutate parameter residency.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
from threading import RLock
import time

import psutil
import torch
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from ..airllm_base import AirLLMBaseModel, restore_relocated_transformers_symbols
from .cache import AdaptiveCache, BoundedPrefetch
from .checkpoint import DirectCheckpoint, atomic_json
from .config import UltraConfig
from .experts import DiskExperts, ExpertSource
from .weights import WeightReader


class AirLLMUltraModel(AirLLMBaseModel):
    def __init__(self, model_local_path_or_repo_id, device=None, dtype=None,
                 ultra=None, **kwargs):
        if kwargs:
            raise TypeError(f'Ultra does not silently ignore options: {sorted(kwargs)}')
        self.ultra = ultra if isinstance(ultra, UltraConfig) else UltraConfig(**(ultra or {}))
        self._lock = RLock()
        self.closed = False
        self.hooks = []
        self.stats = defaultdict(int)
        self.routes = {}
        self.expert_sources = {}
        self._active_names = {}
        self.model_local_path = Path(model_local_path_or_repo_id).resolve()
        if not (self.model_local_path / 'config.json').is_file():
            raise FileNotFoundError('Ultra expects a downloaded local checkpoint directory with config.json; '
                                    'it never starts an implicit giant download')
        self.running_device = device or ('mps' if torch.backends.mps.is_available() else
                                         'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.running_device)
        if self.device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS is not available in this environment')
        self.config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=False,
                                                local_files_only=True)
        self.trust_remote_code = False
        if self.config.model_type not in {'llama', 'mistral', 'mixtral', 'qwen2', 'qwen3',
                                          'qwen3_moe', 'glm_moe_dsa', 'deepseek_v3'}:
            raise NotImplementedError(f'Ultra model_type={self.config.model_type!r} is not validated; '
                                      'use the upstream backend for other families')
        cfg_dtype = getattr(self.config, 'dtype', None) or getattr(self.config, 'torch_dtype', None)
        if isinstance(cfg_dtype, str):
            cfg_dtype = getattr(torch, cfg_dtype)
        self.dtype = dtype or cfg_dtype or torch.float32
        self.running_dtype = self.dtype
        self.hf_quantizer = None
        self.compression = None
        self.set_layer_names_dict()
        restore_relocated_transformers_symbols()
        self.generation_config = self.get_generation_config()
        self.checkpoint = DirectCheckpoint(self.model_local_path,
                                           cache_dir=self.ultra.metadata_dir(),
                                           max_open_files=self.ultra.max_open_files)
        self.checkpoint_path = self.model_local_path
        self.reader = WeightReader(self.checkpoint, self.dtype,
                                   getattr(self.config, 'quantization_config', None))
        self.cache = AdaptiveCache(self.ultra.cache_bytes)
        self.device_cache = AdaptiveCache(self.ultra.device_cache_bytes if self.device.type != 'cpu' else 0)
        self.prefetch = BoundedPrefetch(self.ultra.prefetch_bytes if self.ultra.prefetch else 0)
        self.profile_path = self.ultra.metadata_dir() / (self.checkpoint.fingerprint + '.routes.json')
        self.tokenizer = None
        try:
            # Eager expert implementations have stable routing semantics. Attention
            # uses SDPA when the model supports it, preserving its DSA/cache code.
            self.config._experts_implementation = 'eager'
            self.model = self._instantiate_on_meta('sdpa')
            self.model.eval()
            self.model.requires_grad_(False)
            self.model.tie_weights()
            self.model.generation_config = self.generation_config
            self._patch_device_property()
            self._restore_computed_buffers()
            for name, buf in self.model.named_buffers():
                if buf.device.type != 'meta':
                    set_module_tensor_to_device(self.model, name, self.device, value=buf)
            self._replace_experts()
            self._build_groups()
            self._load_route_profile()
            if self.ultra.load_tokenizer:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_local_path,
                                                               trust_remote_code=False,
                                                               local_files_only=True)
        except BaseException:
            self.close(save_profile=False)
            raise

    def _instantiate_on_meta(self, attn_implementation):
        # init_empty_weights alone moves Parameters to meta AFTER their factory
        # runs. A fused GLM expert bank allocates >12 GiB in torch.empty first.
        # The default-device context prevents that temporary CPU allocation too.
        with torch.device('meta'):
            return super()._instantiate_on_meta(attn_implementation)

    def _restore_computed_buffers(self):
        # Rotary frequencies are derived, non-persistent buffers, absent from
        # the checkpoint. Reconstruct these small modules outside the meta
        # context instead of materializing uninitialized buffer contents.
        for name, module in list(self.model.named_modules()):
            if type(module).__name__.endswith('RotaryEmbedding'):
                rebuilt = type(module)(config=module.config)
                parent, _, attr = name.rpartition('.')
                setattr(self.model.get_submodule(parent), attr, rebuilt)
        # Persistent buffers (notably GLM's learned routing correction) must be
        # loaded from the checkpoint by the same groups as parameters.
        persistent = set(self.model.state_dict())
        for name, buffer in self.model.named_buffers():
            if buffer.device.type == 'meta' and name not in persistent:
                raise NotImplementedError(f'Cannot reconstruct derived meta buffer {name}')

    def _replace_experts(self):
        # New Transformers uses 3D banks; older versions expose ModuleLists.
        for name, module in list(self.model.named_modules()):
            if not name.endswith('.experts'):
                continue
            if hasattr(module, 'gate_up_proj') and hasattr(module, 'down_proj'):
                source = ExpertSource(self, name, module)
                self._check_size(('expert', name), source.size)
                self.expert_sources[name] = source
                parent, _, attr = name.rpartition('.')
                setattr(self.model.get_submodule(parent), attr, DiskExperts(source))
            elif isinstance(module, torch.nn.ModuleList):
                for expert_idx, expert in enumerate(module):
                    prefix = f'{name}.{expert_idx}'
                    keys = self.checkpoint.keys(prefix)
                    if not keys:
                        raise ValueError(f'No checkpoint tensors for {prefix}')
                    key = ('module_expert', name, expert_idx)
                    self._install_group(prefix, expert, keys, key)
                # Exclude this entire bank from its parent layer group.
                self.expert_sources[name] = None

    def _source_key(self, name):
        if name in self.checkpoint.tensors:
            return name
        if self.config.model_type == 'mixtral':
            legacy = name.replace('.mlp.', '.block_sparse_moe.')
            if legacy in self.checkpoint.tensors:
                return legacy
        if getattr(self.config, 'tie_word_embeddings', False):
            aliases = {'lm_head.weight': 'model.embed_tokens.weight',
                       'model.embed_tokens.weight': 'lm_head.weight'}
            if aliases.get(name) in self.checkpoint.tensors:
                return aliases[name]
        raise ValueError(f'Missing checkpoint tensor for model parameter: {name}')

    def _build_groups(self):
        names = ['model.embed_tokens'] + [f'model.layers.{i}' for i in range(len(self.model.model.layers))]
        names += ['model.norm', 'lm_head']
        self.group_names = names
        self.groups = {}
        covered = set()
        for name in names:
            module = self.model.get_submodule(name)
            keys = []
            mapping = {}
            for local_name, _ in module.state_dict(keep_vars=True).items():
                full = name + '.' + local_name
                if any(full.startswith(p + '.') for p in self.expert_sources):
                    continue
                source = self._source_key(full)
                keys.append(source)
                mapping[source] = full
                covered.add(full)
            key = ('layer', name)
            self.groups[key] = (tuple(dict.fromkeys(keys)), mapping)
            self._install_group(name, module, self.groups[key][0], key)
        # Fail closed on any non-expert parameter outside the streaming plan.
        missing = [name for name, _ in self.model.named_parameters(remove_duplicate=False)
                   if name not in covered and not any(name.startswith(p + '.') for p in self.expert_sources)]
        if missing:
            raise NotImplementedError(f'Parameters outside streaming plan: {missing[:8]}')

    def _install_group(self, name, module, keys, key):
        size = self.reader.nbytes(keys)
        self._check_size(key, size)
        def before(mod, args):
            self._memory_pressure()
            state = self.get_device_weights(key, size, lambda: self.reader.load(keys))
            mapping = getattr(self, 'groups', {}).get(key, (None, {}))[1]
            moved = []
            try:
                for source, value in state.items():
                    target = mapping.get(source, source)
                    set_module_tensor_to_device(self.model, target, self.device, value=value,
                                                dtype=value.dtype)
                    moved.append(target)
                self._active_names[key] = moved
            except BaseException:
                for target in moved:
                    set_module_tensor_to_device(self.model, target, 'meta')
                raise
            if key[0] == 'layer':
                idx = self.group_names.index(name)
                if idx + 1 < len(self.group_names):
                    next_name = self.group_names[idx + 1]
                    next_key = ('layer', next_name)
                    next_keys = self.groups[next_key][0]
                    self.prefetch_weights(next_key, self.reader.nbytes(next_keys),
                                          lambda ks=next_keys: self.reader.load(ks))
                    # Persistent routing frequencies are hints only. Actual routing
                    # determines execution and wrong predictions never change output.
                    for prefix, source in self.expert_sources.items():
                        if source is not None and prefix.startswith(next_name + '.'):
                            for expert, _ in self.routes.get(prefix, Counter()).most_common(self.ultra.expert_prediction):
                                self.prefetch_weights(('expert', prefix, expert), source.size,
                                                      lambda e=expert, s=source: s.load(e))
        def after(mod, args, output):
            for target in self._active_names.pop(key, []):
                set_module_tensor_to_device(self.model, target, 'meta')
            return output
        self.hooks.append(module.register_forward_pre_hook(before))
        self.hooks.append(module.register_forward_hook(after, always_call=True))

    def _check_size(self, key, size):
        if size > self.ultra.max_group_bytes:
            raise MemoryError(f'{key}: {size:,} runtime weight bytes exceeds max_group_bytes='
                              f'{self.ultra.max_group_bytes:,}; increase only after checking free memory')

    def _memory_pressure(self):
        self.stats['group_calls'] += 1
        if self.stats['group_calls'] % self.ultra.memory_check_interval:
            return
        available = psutil.virtual_memory().available
        if available < self.ultra.reserve_bytes:
            shortage = self.ultra.reserve_bytes - available
            self.cache.resize(max(0, self.cache.bytes - shortage))
            self.device_cache.resize(max(0, self.device_cache.bytes - shortage))
            self.stats['pressure_shrinks'] += 1

    def get_weights(self, key, size, loader):
        self._check_size(key, size)
        value = self.cache.get(key)
        if value is not None:
            return value
        value = self.prefetch.take(key)
        if value is None:
            start = time.perf_counter()
            value = loader()
            self.stats['sync_load_ns'] += int((time.perf_counter() - start) * 1e9)
        self.cache.put(key, value)
        return value

    def prefetch_weights(self, key, size, loader):
        if self.ultra.prefetch and not self.cache.contains(key) and not self.device_cache.contains(key):
            return self.prefetch.submit(key, size, loader)
        return False

    def get_device_weights(self, key, size, loader):
        if self.device.type == 'cpu':
            return self.get_weights(key, size, loader)
        state = self.device_cache.get(key)
        if state is None:
            host = self.get_weights(key, size, loader)
            # Metal uses unified RAM, but PyTorch CPU/MPS storages are distinct.
            # Keeping hot weights on MPS avoids repeating this real copy.
            state = {k: v.to(self.device) for k, v in host.items()}
            self.device_cache.put(key, state)
            self.stats['device_transfer_bytes'] += sum(v.numel()*v.element_size() for v in state.values())
        return state

    def record_routes(self, prefix, selected):
        counts = self.routes.setdefault(prefix, Counter())
        counts.update(selected)
        if sum(counts.values()) > 1_000_000:
            self.routes[prefix] = Counter({e: max(1, v // 2) for e, v in counts.items()})
        self.stats['expert_calls'] += len(selected)

    def _load_route_profile(self):
        try:
            obj = json.loads(self.profile_path.read_text())
            if obj.get('fingerprint') == self.checkpoint.fingerprint:
                for prefix, counts in obj.get('routes', {}).items():
                    source = self.expert_sources.get(prefix)
                    if source is not None:
                        self.routes[prefix] = Counter({int(e): int(v) for e, v in counts.items()
                                                       if 0 <= int(e) < source.num_experts and int(v) > 0})
        except (ValueError, TypeError, OSError):
            pass

    def metrics(self):
        return {'runtime': dict(self.stats), 'checkpoint': dict(self.checkpoint.stats),
                'cache': {**self.cache.stats, 'bytes': self.cache.bytes,
                          'peak_bytes': self.cache.peak_bytes, 'capacity': self.cache.capacity},
                'device_cache': {**self.device_cache.stats, 'bytes': self.device_cache.bytes,
                                 'peak_bytes': self.device_cache.peak_bytes, 'capacity': self.device_cache.capacity},
                'prefetch': {**self.prefetch.stats, 'bytes': self.prefetch.bytes,
                             'peak_bytes': self.prefetch.peak_bytes, 'budget': self.prefetch.budget},
                'expert_banks': len(self.expert_sources)}

    @torch.inference_mode()
    def generate(self, *args, **kwargs):
        with self._lock:
            if self.closed:
                raise RuntimeError('Model is closed')
            return self.model.generate(*args, **kwargs)

    @torch.inference_mode()
    def forward(self, *args, **kwargs):
        with self._lock:
            if self.closed:
                raise RuntimeError('Model is closed')
            return self.model(*args, **kwargs)

    __call__ = forward

    def close(self, save_profile=True):
        with self._lock:
            if self.closed:
                return
            try:
                self.prefetch.close()
                if save_profile and self.routes:
                    atomic_json(self.profile_path, {'fingerprint': self.checkpoint.fingerprint,
                                                    'routes': self.routes})
            finally:
                for hook in self.hooks:
                    hook.remove()
                self.hooks.clear()
                self.cache.clear()
                self.device_cache.clear()
                self.checkpoint.close()
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
