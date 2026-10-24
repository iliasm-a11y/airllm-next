"""One-instance, batch-one adapter for the existing native Next runtimes.

Logical sessions are durable token transcripts. KV/recurrent arrays are never
serialized: resumption rebuilds them and verifies every committed greedy token.
No model is loaded by this adapter and no existing allocator limit is changed.
"""
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
from threading import get_ident
from time import perf_counter


class NativeModelBackend:
    max_concurrency = 1
    concurrent_safe = False

    def __init__(self, model):
        self.model = model
        self.owner = get_ident()
        self.kv_bytes_per_session = model.options.kv_bytes
        if type(self.kv_bytes_per_session) is not int or self.kv_bytes_per_session < 1:
            raise ValueError('Native adapter requires a positive bounded KV budget')
        checkpoint = model.checkpoint
        fingerprint = getattr(checkpoint, 'fingerprint', None)
        self._gguf_stat = None
        if fingerprint is None:
            from .gguf import GGUFReader
            if isinstance(checkpoint, GGUFReader):
                self._gguf_stat = self._file_identity(checkpoint)
                payload = {'file': self._gguf_stat, 'metadata': checkpoint.metadata,
                           'tensors': [asdict(t) for t in checkpoint.tensors.values()]}
                fingerprint = sha256(json.dumps(payload, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError('A stable local checkpoint fingerprint is required')
        self.checkpoint_fingerprint = fingerprint
        options = (asdict(model.options) if is_dataclass(model.options)
                   else vars(model.options))
        # Include computation options, unlike a model name alone. The reader's
        # fingerprint is metadata based; a full weight hash remains a separate
        # validation step for a benchmark or an untrusted checkpoint directory.
        versions = {}
        for distribution in ('mlx', 'numpy', 'torch'):
            try:
                versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                versions[distribution] = None
        sources = sha256()
        for path in sorted(Path(__file__).parent.glob('*.py')):
            sources.update(path.name.encode())
            sources.update(path.read_bytes())
        identity = dict(checkpoint=fingerprint, config=model.config, versions=versions,
                        native_sources_sha256=sources.hexdigest(),
                        options=options, device=getattr(model, 'device', None),
                        runtime=type(model).__module__+'.'+type(model).__qualname__,
                        mlx_version=getattr(model.mx, '__version__', None),
                        tf32=os.environ.get('MLX_ENABLE_TF32', '1'),
                        adapter_version=1)
        for owner_name, owner in (('model', model), ('reader', getattr(model, 'reader', None)),
                                  ('checkpoint', model.checkpoint)):
            identity[owner_name+'_execution'] = {key: getattr(owner, key, None) for key in
                ('kernel', 'expert_batch', 'concat_tiles', 'pipeline_tiles',
                 'metal_simd_groups', 'packed_copy_free', 'packed_direct_io',
                 'bulk_metadata', 'output_rows', 'staging_bytes', 'cache_decay_tokens')}
        identity['cache_decay_interval'] = getattr(getattr(getattr(model, 'reader', None),
                                                         'cache', None), 'decay_interval', None)
        self.identity = sha256(json.dumps(identity, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self._cache = None
        self._active = None

    @staticmethod
    def _file_identity(checkpoint):
        stat = os.fstat(checkpoint.fileno())
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def validate(self, request):
        if self._gguf_stat is not None and self._file_identity(self.model.checkpoint) != self._gguf_stat:
            raise RuntimeError('The GGUF checkpoint changed after backend construction')
        vocab = self.model.config['vocab_size']
        limit = min(self.model.options.max_context,
                    self.model.config['max_position_embeddings'])
        if (type(request.max_tokens) is not int or request.max_tokens < 1 or
                len(request.output_ids) > request.max_tokens or not request.input_ids or
                len(request.input_ids)+request.max_tokens > limit):
            raise ValueError('Native request exceeds the configured context limit')
        if any(type(t) is not int or not 0 <= t < vocab for t in
               (*request.input_ids, *request.output_ids, *request.eos_token_ids)):
            raise ValueError('Native request contains an invalid token')

    def _predict(self, ids):
        import numpy as np
        output = self.model.forward(np.array([ids], dtype=np.int32),
            past_key_values=self._cache, logits_to_keep=1)
        self._cache = output.past_key_values
        if self._cache is None:
            raise RuntimeError('Native runtime returned no session')
        mx = self.model.mx
        with mx.stream(self.model.stream):
            if not bool(mx.all(mx.isfinite(output.logits)).item()):
                raise FloatingPointError('Non-finite native logits')
            token = int(mx.argmax(output.logits[0, -1], axis=-1).item())
        if not 0 <= token < self.model.config['vocab_size']:
            raise RuntimeError('Invalid native token')
        return token

    def advance(self, request, max_tokens):
        from .orchestrator import InferenceResult
        if get_ident() != self.owner:
            raise RuntimeError('Native inference must run on the model owner thread')
        self.validate(request)
        if self._active is not None:
            raise RuntimeError('The native model already owns an inference')
        self._active = request.job_id
        generated = []
        replay_tokens = 0
        replay_seconds = 0.
        first_token_seconds = None
        peak = 0
        begin = perf_counter()
        try:
            with self.model.lock:
                if getattr(self.model, 'closed', False):
                    raise RuntimeError('Native model is closed')
                if request.output_ids:
                    replay_begin = perf_counter()
                    candidate = None
                    for index, known in enumerate(request.output_ids):
                        if request.stop_event.is_set():
                            return InferenceResult([], False, {'interrupted': True})
                        ids = request.input_ids if index == 0 else [request.output_ids[index-1]]
                        candidate = self._predict(ids)
                        replay_tokens += len(ids)
                        if candidate != known:
                            raise RuntimeError('replay_token_mismatch')
                    replay_seconds = perf_counter()-replay_begin
                for _ in range(min(max_tokens, request.max_tokens-len(request.output_ids))):
                    if request.stop_event.is_set():
                        break
                    previous = (*request.output_ids, *generated)
                    ids = request.input_ids if not previous else [previous[-1]]
                    token = self._predict(ids)
                    if first_token_seconds is None:
                        first_token_seconds = perf_counter()-begin
                    generated.append(token)
                    peak = max(peak, int(getattr(self._cache, 'peak_bytes',
                                                getattr(self._cache, 'bytes', 0))))
                    if token in request.eos_token_ids:
                        break
            finished = (len(request.output_ids)+len(generated) >= request.max_tokens or
                        bool(generated and generated[-1] in request.eos_token_ids))
            return InferenceResult(generated, finished, {
                'replayed_tokens': replay_tokens, 'replay_seconds': replay_seconds,
                'first_token_seconds': first_token_seconds,
                'service_seconds': perf_counter()-begin,
                'resident_context_peak_bytes': peak,
                'interrupted': request.stop_event.is_set()})
        finally:
            self.release(request.job_id)

    def release(self, job_id):
        if get_ident() != self.owner:
            raise RuntimeError('Native sessions must release on the model owner thread')
        if self._active not in (None, job_id):
            raise RuntimeError('Cannot release another native session')
        self._cache = None
        self._active = None
