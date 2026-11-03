# Isolated CPU noBLAS v4 candidate

Status: source and portable scheduling/refusal fixtures only. No helper has been compiled, no native smoke has run, no model or GPU has been started for this version. A/B/C and performance remain unqualified. Original CPU8, v1/v2/v3 sources, libraries, wrappers, comparators and measured receipts are unchanged.

The closed v3 run matched CPU8 through request1 token512, then diverged at its complete542-token state/logits. Its final30-column ordinary Q8 call is eligible for CPU fallback column chunks `ceil(30/ceil(30/16))=15`, so the two-row ARM dot kernel drops to `nrc=1`. This is a source mechanism; the separate component experiment and the future whole-model experiment still need their actual numerical results.

V4 chooses each physical call in this order:

1. Limit by the requested UB, remaining tokens and next absolute fence128/208/256/384/512.
2. If that window is greater than16 and not a multiple16, round down to its preceding multiple16.
3. For an odd remaining window9/11/13/15, use8 first, then the scalar remainder1/3/5/7, matching CPU8's tail. Even14 remains14.

Examples:30→16+14;72→64+8;521 ends8+1;523 ends8+3. This policy applies to every full/split/restore decode, including the prefix, with the same absolute positions. UB8/16/32/64/128 remain experimental choices. The new policy is `cpu8-tail-global-fences208-no-blas-v4`, version4, and explicitly keys `fallback_column_chunk=16` and `fallback_remainder_policy=floor16-then-cpu8-odd-tail`. Each batch records the original fenced window, aligned window and both split causes. No old key is substituted.

`fallback_eligibility.py` models the exact V4 ordinary2D Q8 fallback chunk/thread decisions for a two-row ARM kernel. The observed v3 inventory has ordinary Q8 M values1536/2048/4096/8192/12288/16384/154880, all multiples16 and large enough to avoid the thread-rechunk fallback for N>1 under10 workers. The portable fixtures compare the full per-token/per-row nrc map over all window lengths1..128 and all bounded prompt lengths1..1800. This is eligibility only: K-dependent arithmetic, data, kernel dispatch and full-model fidelity are not proved. Dim2/3 and the source/runtime path must be confirmed in the real inventory. A small-M counterexample is retained: M128 can change its row chunking across N8/N128, so there is deliberately no universal shape claim.

The helper retains v3's actual pre-model BLAS removal and refuses missing BLAS, remaining ACCEL or missing CPU. `--backend-policy-selftest` is run only by a future explicitly authorized build; the build receipt and each actual run must expose the real registry transition. Current source has not observed that transition. Native RSS lifetime API, resource guards, snapshot cap, scratch policy, continuation capture and all unchanged dependencies remain identical.

`variant_compare.py` requires two actual build receipts, actual source/binary/patch/headers/libraries, each own recomputed snapshot key, matching runtime/shards/tokenizer/template/wrapper/snapshot/capture, complete three-request raw captures, exact ordered routes and all CPU bit-exact values, continuations/replays and final/checkpoint states. Only validated helper/backend-policy/batch differences are normalized. An early/probe result cannot qualify it.

`restore_compare.py` additionally reruns the full CPU8→fullV4 comparison before reading restore evidence. It admits the first profile32 only, requires the same actual v4 key/build/registry in full and restore, the exact208 snapshot, matching observed suffix boundaries, every actually prescribed restore checkpoint, raw final/continuation states and metadata bound to each arm's own physical batches. This incorporates the independently reviewed v3 checkpoint-reader correction without modifying the old comparator. Other restore UBs need their own explicitly reviewed profile admission.

Planning from the workspace root (does not compile or start native code):

```sh
CANDIDATE="$PWD/work/native-noblas208-v4-candidate"
V4="/Users/macbook/Documents/Codex/2026-11-02/referenced-chatgpt-conversation-this-is-an-2/work/runtime-v4/bin"
python3 "$CANDIDATE/build_helper.py" \
  --native-source "$PWD/work/metal-integration-v4-source/source" \
  --libraries "$V4" --receipt "$CANDIDATE/build-plan.json"
```

Only after coordinator review, use a fresh `build-receipt.json` and add `--execute`; that compiles the new helper and performs metadata-only parameter/backend smokes. It creates `build-binding.json` only on real success. Candidate binary/build-receipt hashes remain null before that build. `build_helper.py` tracks all explicit ABI/JSON headers before/after compile and after smoke. No existing measured receipt may be edited to claim a new build.

CPU and restore runs must use the unchanged guarded wrapper under the existing UID supervisor, after actual power/resource admission, with a fresh output folder and no concurrent physical engine. This note does not authorize a run. Full comparison commands require both measured control/candidate directories and guards plus `native-prefill-final-controls.build.json`, the new actual `build-receipt.json` and `build-binding.json`; partial captures are refused. A native probe remains explicitly incomplete and can only reject, never qualify.
