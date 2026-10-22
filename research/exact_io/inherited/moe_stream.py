"""Layer-streaming, teacher-forced FP32 forward for open MoE checkpoints (CPU).
One decoder layer is materialised at a time (official transformers / remote-code classes,
weights assigned from safetensors via mmap), applied to the hidden states of every sequence
of the corpus, its router decisions are recorded, then it is freed.
FP8 (e4m3fn) weights with 128x128 'weight_scale_inv' blocks are dequantised: w = fp8->f32 * s."""
import os, sys, json, time, gc, math, resource, importlib
import numpy as np, torch
from safetensors import safe_open

def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6

def cur_rss_gb():
    try:
        import psutil; return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float("nan")

def dequant_fp8(w8, s, B=128):
    w = w8.to(torch.float32)
    o, i = w.shape
    if o % B == 0 and i % B == 0:
        w = w.view(o // B, B, i // B, B).mul_(s.to(torch.float32)[:, None, :, None]).view(o, i)
    else:
        sf = s.to(torch.float32).repeat_interleave(B, 0)[:o].repeat_interleave(B, 1)[:, :i]
        w = w.mul_(sf)
    return w

ST_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8}

class Reader:
    """safetensors reader using explicit positioned reads (no mmap), so that RSS only counts
    the buffers we actually hold. Bit-identical to safe_open (checked in tests)."""
    def __init__(self, d):
        self.d = d
        self.wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}
        self.bytes_read = 0
    def _hdr(self, shard):
        if shard not in self.hdr:
            p = os.path.join(self.d, shard)
            with open(p, "rb") as f:
                n = int.from_bytes(f.read(8), "little"); h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            self.hdr[shard] = (p, 8 + n, h)
        return self.hdr[shard]
    def has(self, k): return k in self.wm
    def raw(self, k):
        p, base, h = self._hdr(self.wm[k])
        m = h[k]; s0, s1 = m["data_offsets"]
        buf = bytearray(s1 - s0)
        with open(p, "rb", buffering=0) as f:
            f.seek(base + s0); mv = memoryview(buf); got = 0
            while got < len(buf):
                r = f.readinto(mv[got:]); assert r, "short read"; got += r
        self.bytes_read += len(buf)
        t = torch.frombuffer(buf, dtype=ST_DT[m["dtype"]]) if len(buf) else torch.empty(0, dtype=ST_DT[m["dtype"]])
        return t.reshape(m["shape"])
    def f32(self, k):
        t = self.raw(k)
        if t.dtype == torch.float8_e4m3fn:
            return dequant_fp8(t, self.raw(k + "_scale_inv"))
        return t.to(torch.float32)
    def close(self):
        self.hdr.clear()

# ----------------------------------------------------------------------------- routers
def select_topk(family, logits, k, bias=None, n_group=1, topk_group=1, norm_topk=False, scale=1.0, kk=None):
    """Reproduces the selection rule of each family. Returns (ids[T,k], gate_w[T,k], cand_ids[T,kk], cand_scores[T,kk])."""
    kk = kk or 2 * k
    logits = logits.float()
    if family in ("olmoe", "qwen3_moe"):
        p = torch.softmax(logits, dim=-1)
        cs, ci = torch.topk(p, kk, dim=-1)
        w = cs[:, :k].clone()
        if norm_topk:
            w = w / w.sum(-1, keepdim=True)
        return ci[:, :k], w * scale, ci, cs
    if family == "ling":
        s = torch.sigmoid(logits)
        sc = s + bias[None, :] if bias is not None else s
        T, N = sc.shape
        if n_group > 1:
            g = sc.view(T, n_group, N // n_group).topk(2, dim=-1)[0].sum(-1)
            gi = torch.topk(g, topk_group, dim=-1)[1]
            gm = torch.zeros_like(g).scatter_(1, gi, 1.0)
            m = gm.unsqueeze(-1).expand(T, n_group, N // n_group).reshape(T, N)
            sc = sc.masked_fill(m == 0, float("-inf"))
        cs, ci = torch.topk(sc, kk, dim=-1)
        w = s.gather(1, ci[:, :k])
        if norm_topk:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return ci[:, :k], w * scale, ci, cs
    raise ValueError(family)

# ----------------------------------------------------------------------------- family adapters
class HFAdapter:
    """olmoe / qwen3_moe: native transformers classes; experts fused as [N,2I,H] / [N,H,I]."""
    def __init__(self, family, model_dir):
        from transformers import AutoConfig
        self.family = family
        self.cfg = AutoConfig.from_pretrained(model_dir)
        self.cfg._attn_implementation = "sdpa"
        self.cfg._experts_implementation = "eager"
        mod = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
        pre = {"olmoe": "Olmoe", "qwen3_moe": "Qwen3Moe"}[family]
        self.Layer = getattr(mod, pre + "DecoderLayer")
        self.Rotary = getattr(mod, pre + "RotaryEmbedding")
        c = self.cfg
        self.N = getattr(c, "num_experts", None) or c.num_local_experts
        self.k = c.num_experts_per_tok
        self.L = c.num_hidden_layers
        self.norm_topk = bool(getattr(c, "norm_topk_prob", False))
        self.eps = c.rms_norm_eps
        sparse = []
        for i in range(self.L):
            if family == "qwen3_moe":
                mlp_only = getattr(c, "mlp_only_layers", []) or []
                step = getattr(c, "decoder_sparse_step", 1)
                sparse.append(i not in mlp_only and (i + 1) % step == 0)
            else:
                sparse.append(True)
        self.sparse = sparse
    def moe_layers(self): return [i for i in range(self.L) if self.sparse[i]]
    def router_weight_key(self, i): return f"model.layers.{i}.mlp.gate.weight"
    def router_norm_key(self, i): return f"model.layers.{i}.post_attention_layernorm.weight"
    def router_bias(self, R, i): return None
    def select(self, logits, bias=None):
        return select_topk(self.family, logits, self.k, norm_topk=self.norm_topk)
    def rmsnorm(self, x, w):
        xf = x.float(); v = xf.pow(2).mean(-1, keepdim=True)
        return w * (xf * torch.rsqrt(v + self.eps))
    def build_layer(self, R, i):
        with torch.device("meta"):
            layer = self.Layer(self.cfg, i)
        sd = {}
        p = f"model.layers.{i}."
        names = list(layer.state_dict().keys())
        for n in names:
            if n.startswith("mlp.experts."):
                continue
            sd[n] = R.f32(p + n)
        if self.sparse[i]:
            N = self.N
            g0 = R.f32(p + "mlp.experts.0.gate_proj.weight")
            I, H = g0.shape
            gu = torch.empty(N, 2 * I, H); dn = torch.empty(N, H, I)
            for e in range(N):
                q = p + f"mlp.experts.{e}."
                gu[e, :I] = g0 if e == 0 else R.f32(q + "gate_proj.weight")
                gu[e, I:] = R.f32(q + "up_proj.weight")
                dn[e] = R.f32(q + "down_proj.weight")
            del g0
            sd["mlp.experts.gate_up_proj"] = gu; sd["mlp.experts.down_proj"] = dn
        missing = set(names) - set(sd)
        assert not missing, missing
        layer.load_state_dict(sd, strict=True, assign=True)
        layer.eval()
        return layer
    def gate_module(self, layer): return layer.mlp.gate
    def gate_hook_parse(self, out):
        logits, scores, idx = out
        return logits
    def gate_hook_ids(self, out):
        logits, scores, idx = out
        return idx, scores
    def postnorm_module(self, layer): return layer.post_attention_layernorm
    def call_layer(self, layer, h, cos_sin, pos):
        out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=cos_sin)
        return out[0] if isinstance(out, tuple) else out
    def embed_key(self): return "model.embed_tokens.weight"
    def final_norm_key(self): return "model.norm.weight"
    def lm_head_key(self, R):
        return "lm_head.weight" if R.has("lm_head.weight") else "model.embed_tokens.weight"
    def rotary(self):
        return self.Rotary(self.cfg)

def get_adapter(family, model_dir):
    if family in ("olmoe", "qwen3_moe"):
        return HFAdapter(family, model_dir)
    if family == "ling":
        import ling_adapter
        return ling_adapter.LingAdapter(model_dir)
    raise ValueError(family)

# ----------------------------------------------------------------------------- streaming forward
@torch.inference_mode()
def stream_forward(A, R, seqs, record=True, lookahead=True, layers=None, log=print, keep_hidden=None, chunk_tokens=4096):
    """seqs: list of LongTensor batches [B, T] (no padding). Each layer is built once, applied to every
    batch in chunks of <= chunk_tokens tokens; router decisions are taken from the router module's own
    outputs (ground truth) and cross-checked against an independent re-implementation of the rule."""
    L = A.L if layers is None else layers
    moe = [i for i in range(L) if A.sparse[i]]
    E = R.f32(A.embed_key())
    Hs = [E[s] for s in seqs]
    del E; gc.collect()
    rot = A.rotary()
    cs = {}
    for s in seqs:
        T = s.shape[1]
        if T not in cs:
            pos = torch.arange(T).unsqueeze(0)
            cs[T] = (rot(Hs[0][:1, :1], pos), pos)
    Wr, Wn, Bs = {}, {}, {}
    for i in [j for j in range(A.L) if A.sparse[j]]:
        if i < L + 2:
            Wr[i] = R.f32(A.router_weight_key(i)); Wn[i] = R.f32(A.router_norm_key(i)); Bs[i] = A.router_bias(R, i)
    rec, kept = {}, {}
    for i in range(L):
        t0 = time.time(); b0 = R.bytes_read
        layer = A.build_layer(R, i)
        t_load = time.time() - t0
        cap, cap_mid = [], []
        hk = []
        if A.sparse[i]:
            hk.append(A.gate_module(layer).register_forward_hook(lambda m, a, o: cap.append(o)))
            if lookahead:
                hk.append(A.postnorm_module(layer).register_forward_pre_hook(lambda m, a: cap_mid.append(a[0].detach().reshape(-1, a[0].shape[-1]))))
        acc = {"ids": [], "w": [], "cand_ids": [], "cand_scores": [], 0: [], 1: [], 2: []}
        nmis = 0; ntok = 0
        for b, s in enumerate(seqs):
            h = Hs[b]; T = h.shape[1]
            (cos_sin, pos) = cs[T]
            chunk = max(1, chunk_tokens // T)
            outs = []
            for c in range(0, h.shape[0], chunk):
                hc = h[c:c + chunk]
                cap.clear(); cap_mid.clear()
                outs.append(A.call_layer(layer, hc, cos_sin, pos))
                if A.sparse[i] and record:
                    assert len(cap) == 1, len(cap)
                    lg = A.gate_hook_parse(cap[0]).detach().reshape(-1, A.N).float()
                    mid_ids, mod_w = A.gate_hook_ids(cap[0])
                    ids, w, ci, cs_ = A.select(lg, Bs[i])
                    same = (torch.sort(ids, -1)[0] == torch.sort(mid_ids.reshape(ids.shape).long(), -1)[0]).all(-1)
                    nmis += int((~same).sum()); ntok += ids.shape[0]
                    acc["ids"].append(mid_ids.reshape(ids.shape).to(torch.int16).numpy())
                    acc["w"].append(mod_w.reshape(w.shape).float().numpy().astype(np.float16))
                    acc["cand_ids"].append(ci.to(torch.int16).numpy()); acc["cand_scores"].append(cs_.float().numpy().astype(np.float16))
                    if lookahead:
                        assert len(cap_mid) == 1
                        src0 = hc.reshape(-1, hc.shape[-1]); mid = cap_mid[0]
                        for d, src in [(0, src0), (1, mid), (2, mid)]:
                            j = i + d
                            if j in Wr:
                                lgp = A.rmsnorm(src, Wn[j]) @ Wr[j].T
                                _, _, pci, _ = A.select(lgp, Bs[j])
                                acc[d].append(pci.to(torch.int16).numpy())
                cap.clear(); cap_mid.clear()
            Hs[b] = torch.cat(outs, 0)
        for x in hk: x.remove()
        if keep_hidden is not None:
            kept[i] = [x.clone() for x in Hs]
        if A.sparse[i] and record:
            r = {k2: np.concatenate(acc[k2], 0) for k2 in ("ids", "w", "cand_ids", "cand_scores")}
            r["pred"] = {d: np.concatenate(acc[d], 0) for d in (0, 1, 2) if acc[d]}
            r["reimpl_mismatch_tokens"] = nmis; r["n_tokens"] = ntok
            rec[i] = r
        del layer, acc
        gc.collect()
        log(f"layer {i:3d} load {t_load:5.1f}s total {time.time() - t0:6.1f}s read {(R.bytes_read - b0) / 1e9:5.2f}GB rss {cur_rss_gb():5.2f}GB peak {peak_rss_gb():5.2f}GB" + (f" reimpl_mismatch {nmis}/{ntok}" if A.sparse[i] else ""))
    return {"rec": rec, "H": Hs, "kept": kept}

@torch.inference_mode()
def lm_eval(A, R, Hs, seqs, chunk=1024):
    """final norm + lm_head -> next-token CE (nats) and top-1 accuracy per position (last position = nan)."""
    wn = R.f32(A.final_norm_key()); W = R.f32(A.lm_head_key(R))
    out = []
    for h, s in zip(Hs, seqs):
        B, T, Hd = h.shape
        x = A.final_norm(h, wn) if hasattr(A, "final_norm") else A.rmsnorm(h, wn)
        x = x.reshape(-1, Hd); tgt = torch.full((B, T), -1, dtype=torch.long); tgt[:, :-1] = s[:, 1:]; tgt = tgt.reshape(-1)
        ce = torch.full((B * T,), float("nan")); acc = torch.full((B * T,), float("nan"))
        for c in range(0, B * T, chunk):
            lg = x[c:c + chunk] @ W.T
            lp = torch.log_softmax(lg, -1)
            t = tgt[c:c + chunk]; ok = t >= 0
            ce[c:c + chunk][ok] = -lp[ok, t[ok]]
            acc[c:c + chunk][ok] = (lg[ok].argmax(-1) == t[ok]).float()
        out.append((ce.view(B, T).numpy(), acc.view(B, T).numpy()))
    del W
    return out
