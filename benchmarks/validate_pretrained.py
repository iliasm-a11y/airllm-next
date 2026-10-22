#!/usr/bin/env python3
"""Optional trained-model check. Download is explicit and bounded to ~270 MB.

The distributed archive contains no model weights. Metadata pins the reference
model revision; local directories can be checked without any network access.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'air_llm'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir')
    p.add_argument('--download-dir')
    p.add_argument('--metadata', default='evidence/trained-model-metadata.json')
    p.add_argument('--output', default='benchmarks/results-pretrained.json')
    args = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from airllm.ultra import AirLLMUltraModel, UltraConfig
    torch.set_num_threads(1)
    metadata = json.loads(Path(args.metadata).read_text())
    if args.model_dir:
        root = Path(args.model_dir)
    elif args.download_dir:
        from huggingface_hub import snapshot_download
        selected = [f for f in metadata['files'] if f['name'].endswith(('.json', '.txt', '.model', '.safetensors'))]
        if sum(f['bytes'] or 0 for f in selected) > 350_000_000:
            raise ValueError('Refusing a download larger than 350 MB')
        root = Path(snapshot_download(metadata['model_id'], revision=metadata['revision'],
                                      local_dir=args.download_dir, allow_patterns=[f['name'] for f in selected]))
    else:
        p.error('Pass --model-dir for local validation or --download-dir for explicit download')
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    text = tok.apply_chat_template([{'role': 'user', 'content': 'What is the capital of France?'}],
                                  tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors='pt')['input_ids']
    with torch.inference_mode():
        ref = AutoModelForCausalLM.from_pretrained(root, local_files_only=True, dtype=torch.float32).eval()
        expected = ref(ids, use_cache=False, logits_to_keep=1).logits
        expected_ids = ref.generate(ids, max_new_tokens=8, do_sample=False)
        del ref
        start = time.perf_counter()
        cfg = UltraConfig(cache_bytes=128*1024**2, device_cache_bytes=0, prefetch_bytes=32*1024**2,
                          max_group_bytes=256*1024**2, reserve_bytes=0,
                          cache_dir=str(root / '.ultra-metadata'), load_tokenizer=False)
        with AirLLMUltraModel(root, device='cpu', dtype=torch.float32, ultra=cfg) as model:
            construction = time.perf_counter()-start
            begin = time.perf_counter()
            actual = model(ids, use_cache=False, logits_to_keep=1).logits
            forward_seconds = time.perf_counter()-begin
            torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
            actual_ids = model.generate(ids, max_new_tokens=8, do_sample=False)
            assert torch.equal(actual_ids, expected_ids)
            result = {'model_id': metadata['model_id'], 'revision': metadata['revision'],
                      'torch': torch.__version__, 'device': 'cpu', 'dtype': 'float32',
                      'max_absolute_logit_difference': (actual-expected).abs().max().item(),
                      'generated_token_ids_match_reference': True,
                      'output': tok.decode(actual_ids[0][ids.shape[1]:], skip_special_tokens=True),
                      'input_tokens': ids.shape[1], 'new_tokens': actual_ids.shape[1]-ids.shape[1],
                      'construction_seconds': construction, 'first_forward_seconds': forward_seconds,
                      'metrics': model.metrics(),
                      'limitation': 'One small trained Llama-family model on CPU; not a GLM-5.3/M5 validation.'}
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
