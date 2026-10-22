import os, sys, json, time, argparse, gc
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_stream as ms, corpus_tok

ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True); ap.add_argument("--model_dir", required=True)
ap.add_argument("--name", required=True); ap.add_argument("--mode", default="validate")
ap.add_argument("--threads", type=int, default=6); ap.add_argument("--nval", type=int, default=2)
ap.add_argument("--vlen", type=int, default=96)
args = ap.parse_args()
torch.set_num_threads(args.threads)
os.makedirs("out", exist_ok=True)
logf = open(f"out/{args.name}_{args.mode}.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
A = ms.get_adapter(args.family, args.model_dir)
R = ms.Reader(args.model_dir)
fam = args.family
if fam == "olmoe":
    prefix, sep = [tok.eos_token_id], [tok.eos_token_id]
elif fam == "qwen3_moe":
    e = tok.convert_tokens_to_ids("<|endoftext|>"); prefix, sep = [], [e]
else:
    prefix, sep = A.doc_prefix(tok), A.doc_sep(tok)
spec = json.load(open("corpus/corpus_spec.json"))
corp = corpus_tok.build(tok, spec, prefix, sep)
cats = list(corp["long"].keys())
log(f"== {args.name} family={fam} L={A.L} N={A.N} k={A.k} moe_layers={len(A.moe_layers())} prefix={prefix} sep={sep} threads={args.threads}")

if args.mode == "validate":
    # reference = official from_pretrained model truncated to nval layers, FP32, sdpa
    from transformers import AutoModelForCausalLM
    nval = args.nval
    ids = torch.tensor(np.stack([corp["long"]["en_prose"]["ids"][:args.vlen], corp["long"]["code"]["ids"][:args.vlen]]), dtype=torch.long)
    t0 = time.time()
    ref = A.load_reference(nval) if hasattr(A, "load_reference") else AutoModelForCausalLM.from_pretrained(args.model_dir, num_hidden_layers=nval, dtype=torch.float32, attn_implementation="sdpa", trust_remote_code=True)
    ref.eval()
    log(f"reference loaded {time.time() - t0:.1f}s rss {ms.cur_rss_gb():.2f}GB class={type(ref).__name__}")
    outs, gates = {}, {}
    layers = ref.model.layers
    hk = []
    for i, l in enumerate(layers):
        hk.append(l.register_forward_hook(lambda m, a, o, i=i: outs.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach().clone())))
        if A.sparse[i]:
            g = A.gate_module(l)
            hk.append(g.register_forward_hook(lambda m, a, o, i=i: gates.__setitem__(i, A.gate_hook_parse(o).detach().reshape(-1, A.N).float().clone())))
    with torch.inference_mode():
        ref(input_ids=ids, use_cache=False)
    for x in hk: x.remove()
    del ref; gc.collect()
    res = ms.stream_forward(A, R, [ids], layers=nval, keep_hidden=True, lookahead=False, log=log)
    report = {"model": args.name, "input_shape": list(ids.shape), "layers": []}
    for i in range(nval):
        a = outs[i]; b = res["kept"][i][0]
        d = (a - b).abs()
        row = {"layer": i, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "ref_max_abs": float(a.abs().max()), "rel_max": float(d.max() / a.abs().max())}
        if A.sparse[i]:
            ida, _, _, _ = A.select(gates[i], A.router_bias(R, i))
            idb = torch.tensor(res["rec"][i]["ids"]).long()
            same_set = np.mean([set(x.tolist()) == set(y.tolist()) for x, y in zip(ida, idb)])
            row.update({"router_logit_max_abs_diff": None, "topk_set_agreement": float(same_set)})
        report["layers"].append(row); log(json.dumps(row))
    report["peak_rss_gb"] = ms.peak_rss_gb()
    json.dump(report, open(f"out/{args.name}_validation.json", "w"), indent=1)
    log("VALIDATION DONE", json.dumps(report)[:300])
    sys.exit(0)

# ------------------------------------------------------------------ full run
seqs = [torch.tensor(corp["long"][c]["ids"][None], dtype=torch.long) for c in cats]
seqs.append(torch.tensor(corp["short"]["ids"], dtype=torch.long))
t0 = time.time()
res = ms.stream_forward(A, R, seqs, layers=None, lookahead=True, log=log)
log(f"forward done {time.time() - t0:.0f}s")
ev = ms.lm_eval(A, R, res["H"], seqs)
moe = A.moe_layers(); k = A.k
nL = len(cats); TL = seqs[0].shape[1]; nS, TS = seqs[-1].shape
def split(arr, last):
    a = np.asarray(arr)
    lo = a[: nL * TL].reshape(nL, TL, last); sh = a[nL * TL:].reshape(nS, TS, last)
    return lo, sh
dt = np.uint8 if A.N <= 256 else np.int16
core, aux = {}, {}
ids_l, ids_s, w_l, w_s, ci_l, ci_s, cs_l, cs_s = [], [], [], [], [], [], [], []
pred = {0: [], 1: [], 2: []}
for i in moe:
    r = res["rec"][i]
    a, b = split(r["ids"], k); ids_l.append(a); ids_s.append(b)
    a, b = split(r["w"], k); w_l.append(a); w_s.append(b)
    a, b = split(r["cand_ids"], r["cand_ids"].shape[1]); ci_l.append(a); ci_s.append(b)
    a, b = split(r["cand_scores"], r["cand_scores"].shape[1]); cs_l.append(a); cs_s.append(b)
    for d in (0, 1, 2):
        p = r["pred"].get(d)
        pred[d].append(p if p is not None else -np.ones((nL * TL + nS * TS, 2 * k), np.int16))
core = dict(moe_layers=np.array(moe), N=A.N, k=k, cats_long=np.array(cats), cats_short=np.array(corp["short"]["cats"]),
            tokens_long=np.stack([corp["long"][c]["ids"] for c in cats]), tokens_short=corp["short"]["ids"],
            assistant_mask_long=np.stack([corp["long"][c]["assistant_mask"] for c in cats]),
            ids_long=np.stack(ids_l).astype(dt), w_long=np.stack(w_l), ids_short=np.stack(ids_s).astype(dt), w_short=np.stack(w_s),
            ce_long=np.concatenate([e[0] for e in ev[:-1]], 0).astype(np.float32), acc_long=np.concatenate([e[1] for e in ev[:-1]], 0).astype(np.float32),
            ce_short=ev[-1][0].astype(np.float32), acc_short=ev[-1][1].astype(np.float32))
aux = dict(cand_ids_long=np.stack(ci_l).astype(dt), cand_scores_long=np.stack(cs_l), cand_ids_short=np.stack(ci_s).astype(dt), cand_scores_short=np.stack(cs_s),
           pred_d0=np.stack(pred[0]).astype(np.int16), pred_d1=np.stack(pred[1]).astype(np.int16), pred_d2=np.stack(pred[2]).astype(np.int16))
np.savez_compressed(f"out/routing_traces_{args.name}.npz", **core)
np.savez_compressed(f"out/routing_aux_{args.name}.npz", **aux)
ce = core["ce_long"]
summ = {"model": args.name, "ce_long_per_cat": {c: float(np.nanmean(ce[j])) for j, c in enumerate(cats)},
        "acc_long_per_cat": {c: float(np.nanmean(core["acc_long"][j])) for j, c in enumerate(cats)},
        "ce_short_mean": float(np.nanmean(core["ce_short"])), "forward_s": time.time() - t0, "peak_rss_gb": ms.peak_rss_gb(),
        "bytes_read_GB": R.bytes_read / 1e9, "n_tokens": int(nL * TL + nS * TS)}
json.dump(summ, open(f"out/{args.name}_run_summary.json", "w"), indent=1)
log("RUN DONE", json.dumps(summ))
