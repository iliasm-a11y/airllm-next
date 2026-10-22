"""Explicit, conservative budgets for out-of-core inference (bytes, not model size)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class UltraConfig:
    cache_bytes: int = 512 * 1024**2
    device_cache_bytes: int = 256 * 1024**2
    prefetch_bytes: int = 256 * 1024**2
    max_group_bytes: int = 2 * 1024**3
    reserve_bytes: int = 2 * 1024**3
    max_open_files: int = 8
    prefetch: bool = True
    expert_prediction: int = 2
    cache_dir: Optional[str] = None
    memory_check_interval: int = 32
    load_tokenizer: bool = True

    def __post_init__(self):
        for key in ('cache_bytes', 'device_cache_bytes', 'prefetch_bytes', 'reserve_bytes', 'expert_prediction'):
            if getattr(self, key) < 0:
                raise ValueError(f'{key} must be nonnegative')
        for key in ('max_group_bytes', 'max_open_files', 'memory_check_interval'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')

    def metadata_dir(self):
        # Keep small metadata off the external SSD by default; never copy weights here.
        return Path(self.cache_dir or Path.home() / '.cache' / 'airllm-ultra')
