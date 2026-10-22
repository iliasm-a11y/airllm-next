"""Route-aware expert execution, for both 3D and per-expert checkpoints.

Unlike a hook on a fused expert bank, this never materializes all experts. The
real router/attention/KV cache remain owned by Transformers. Only the expert
matrix multiplications are replaced, preserving its eager scatter semantics.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ExpertSource:
    def __init__(self, owner, prefix, original):
        self.owner = owner
        self.prefix = prefix
        self.num_experts = original.num_experts
        self.act_fn = original.act_fn
        cp = owner.checkpoint
        storage_prefix = prefix
        if owner.config.model_type == 'mixtral' and not cp.keys(prefix):
            storage_prefix = prefix.replace('.mlp.', '.block_sparse_moe.')
        self.storage_prefix = storage_prefix
        self.fused = storage_prefix + '.gate_up_proj' in cp.tensors
        if self.fused:
            self.keys = (storage_prefix + '.gate_up_proj', storage_prefix + '.down_proj')
            for key in self.keys:
                d = cp.tensors[key]
                if len(d['shape']) != 3 or d['shape'][0] != self.num_experts:
                    raise ValueError(f'Unrecognized fused expert layout: {key}')
                if d['dtype'].startswith('F8_'):
                    raise NotImplementedError('3D FP8 banks are not supported; per-expert FP8 is supported')
            self.size = sum(math.prod(cp.tensors[k]['shape'][1:]) for k in self.keys)
            self.size *= torch.empty((), dtype=owner.dtype).element_size()
        else:
            self.per_expert = {}
            candidates = [('gate_proj', 'up_proj', 'down_proj'), ('w1', 'w3', 'w2')]
            for expert in range(self.num_experts):
                found = None
                for names in candidates:
                    keys = tuple(f'{storage_prefix}.{expert}.{n}.weight' for n in names)
                    if all(k in cp.tensors for k in keys):
                        found = keys
                        break
                if found is None:
                    raise NotImplementedError(f'No recognized weights for {prefix} expert {expert}')
                self.per_expert[expert] = found
            self.size = max(owner.reader.nbytes(keys) for keys in self.per_expert.values())

    def load(self, expert):
        if self.fused:
            cp = self.owner.checkpoint
            return {'gate_up': cp.read_expert_slice(self.keys[0], expert, self.owner.dtype),
                    'down': cp.read_expert_slice(self.keys[1], expert, self.owner.dtype)}
        keys = self.per_expert[expert]
        state = self.owner.reader.load(keys)
        return dict(zip(('gate', 'up', 'down'), (state[k] for k in keys)))


class DiskExperts(nn.Module):
    def __init__(self, source):
        super().__init__()
        # Avoid registering the owner model as a child (and creating a cycle).
        object.__setattr__(self, 'source', source)
        self.num_experts = source.num_experts
        self.act_fn = source.act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        source = self.source
        owner = source.owner
        # One host sync for the whole routed set, not one .item() per expert.
        selected = torch.unique(top_k_index).detach().cpu().tolist()
        selected = [int(e) for e in selected if int(e) != self.num_experts]
        if any(e < 0 or e >= self.num_experts for e in selected):
            raise ValueError('Router returned an invalid expert id')
        owner.record_routes(source.prefix, selected)
        result = torch.zeros_like(hidden_states)
        for pos, expert in enumerate(selected):
            key = ('expert', source.prefix, expert)
            weights = owner.get_device_weights(key, source.size, lambda e=expert: source.load(e))
            # Routing is known now: overlap reading the following ACTUAL expert.
            if pos + 1 < len(selected):
                nxt = selected[pos + 1]
                owner.prefetch_weights(('expert', source.prefix, nxt), source.size,
                                       lambda e=nxt: source.load(e))
            token_idx, top_pos = torch.where(top_k_index == expert)
            current = hidden_states[token_idx]
            device_weights = weights
            if 'gate_up' in device_weights:
                gate, up = F.linear(current, device_weights['gate_up']).chunk(2, dim=-1)
            else:
                gate = F.linear(current, device_weights['gate'])
                up = F.linear(current, device_weights['up'])
            value = F.linear(self.act_fn(gate) * up, device_weights['down'])
            value *= top_k_weights[token_idx, top_pos, None]
            result.index_add_(0, token_idx, value.to(result.dtype))
            # CPU cache owns weights; accelerator temporaries die after this expert.
            del device_weights, weights
        return result
