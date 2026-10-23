from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

GiB = 1024 ** 3
MiB = 1024 ** 2


@dataclass(frozen=True)
class NextConfig:
    """Weight/KV/IO budgets are disjoint; MLX allocator headroom is reserved too."""
    cache_bytes: int = 2 * GiB
    # A partition of cache_bytes, never an additional allocation allowance.
    compact_cache_bytes: int = 0
    output_tile_cache: bool = False
    output_prefetch: bool = False
    io_bytes: int = 256 * MiB
    kv_bytes: int = 512 * MiB
    workspace_bytes: int = 768 * MiB
    reserve_bytes: int = 4 * GiB
    max_tensor_bytes: int = 512 * MiB
    io_workers: int = 2
    max_open_files: int = 16
    row_chunk: int = 2048
    prefill_chunk: int = 16
    max_context: int = 4096
    policy: Literal['lru', 'slru', 'lfu', 'cost', 'adaptive', 'shared', 'expert', 'hybrid', 'batch-aware'] = 'slru'
    device: Literal['auto', 'cpu', 'gpu'] = 'auto'
    dtype: Literal['float32', 'bfloat16', 'float16'] = 'bfloat16'
    profile: Literal['balanced', 'performance'] = 'balanced'
    prefetch: bool = True
    # Experimental predictions share io_bytes; zero selects a conservative cap.
    spec_prefetch: bool = False
    spec_prefetch_bytes: int = 0
    spec_prefetch_experts: int = 1
    io_mode: Literal['pread', 'mmap', 'pread_nocache'] = 'pread'
    profile_io: bool = False
    coalesce_reads: bool = False
    selective_fp8_scales: bool = True
    cache_dir: str | None = None
    quantized_dir: str | None = None
    lossless_dir: str | None = None
    lossless_native_library: str | None = None
    lossless_verify_hash: bool = True
    load_tokenizer: bool = True
    use_saved_tuning: bool = True

    def __post_init__(self):
        for k in ('cache_bytes','compact_cache_bytes','io_bytes','kv_bytes','workspace_bytes','reserve_bytes','spec_prefetch_bytes'):
            if type(getattr(self,k)) is not int or getattr(self,k) < 0:
                raise ValueError(f'{k} must be a nonnegative integer')
        for k in ('max_tensor_bytes','io_workers','max_open_files','row_chunk','prefill_chunk','max_context','spec_prefetch_experts'):
            if type(getattr(self,k)) is not int or getattr(self,k) < 1:
                raise ValueError(f'{k} must be a positive integer')
        for k in ('prefetch','spec_prefetch','profile_io','coalesce_reads','selective_fp8_scales','lossless_verify_hash','output_tile_cache','output_prefetch',
                  'load_tokenizer','use_saved_tuning'):
            if type(getattr(self,k)) is not bool:
                raise ValueError(f'{k} must be boolean')
        if self.policy not in ('lru','slru','lfu','cost','adaptive','shared','expert','hybrid','batch-aware'): raise ValueError('Unknown cache policy')
        if self.io_mode not in ('pread','mmap','pread_nocache'): raise ValueError('Unknown I/O mode')
        if self.spec_prefetch_bytes > self.io_bytes:
            raise ValueError('spec_prefetch_bytes must fit inside io_bytes')
        if self.spec_prefetch_experts > 64:
            raise ValueError('spec_prefetch_experts must be at most 64')
        if self.device not in ('auto','cpu','gpu'): raise ValueError('Unknown compute device')
        if self.dtype not in ('float32','bfloat16','float16'): raise ValueError('Unknown compute dtype')
        if self.profile not in ('balanced','performance'): raise ValueError('Unknown profile')
        if self.compact_cache_bytes > self.cache_bytes:
            raise ValueError('compact_cache_bytes must fit inside cache_bytes')
        if self.lossless_dir and self.io_mode == 'pread_nocache':
            raise ValueError('pread_nocache is not supported for compressed lossless transport')
        if self.lossless_dir and self.quantized_dir:
            raise ValueError('Lossless original-weight transport cannot be combined with a quantized sidecar')
        if self.lossless_native_library and not self.lossless_dir:
            raise ValueError('A native palette decoder requires lossless_dir')

    def metadata_dir(self):
        return Path(self.cache_dir or Path.home()/'.cache'/'airllm-next')

    @property
    def spec_prefetch_budget(self):
        return self.spec_prefetch_bytes or min(32 * MiB, self.io_bytes // 4)

    @property
    def mlx_memory_bytes(self):
        # Compact NumPy buffers live outside MLX's allocator. Their reserved
        # bytes must not also be offered to the MLX allocation ceiling.
        return self.cache_bytes-self.compact_cache_bytes+self.io_bytes+self.kv_bytes+self.workspace_bytes

    def as_dict(self): return asdict(self)
