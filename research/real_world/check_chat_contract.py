#!/usr/bin/env python3
"""Audit metadata-only GGUF chat rendering against local official assets.

No model weights, inference runtime or GPU are loaded. The optional online check
downloads only the small official tokenizer/generation configurations, never
tokenizer vocabularies or weights. Local reference assets must already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'air_llm'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference-tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-ref', default='ac93b5d')
    parser.add_argument('--verify-small-official-configs-online', action='store_true')
    args = parser.parse_args()

    from airllm.nextgen.gguf import GGUFReader
    from airllm.nextgen.gguf_tokenizer import GGUFTokenizer
    # Tokenizer-only audit: avoid initializing Torch and its optional backends.
    os.environ['USE_TORCH'] = '0'
    os.environ['USE_TF'] = '0'
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    result = {'kind': 'metadata_only_official_chat_contract_audit',
              'model': str(args.model), 'reference_assets': str(args.reference_tokenizer),
              'versions': {'python': platform.python_version(), 'tokenizers': tokenizers.__version__,
                           'transformers': transformers.__version__},
              'source_sha256': digest(ROOT / 'air_llm/airllm/nextgen/gguf_tokenizer.py'),
              'reference_sha256': {n: digest(args.reference_tokenizer / n)
                                   for n in ('tokenizer.json', 'tokenizer_config.json')},
              'baseline_ref': args.baseline_ref, 'weight_reads': False, 'gpu_inference': False,
              'official_tokenization_reference': 'tokenizers.Tokenizer.from_file: unmodified official tokenizer.json',
              'chat_rendering_reference': 'AutoTokenizer applies unmodified official tokenizer_config.json Jinja template',
              'transformers_encoding_caveat': 'Qwen2Tokenizer in transformers 5.20.2 replaces the official pretokenizer; compare separately'}
    if args.verify_small_official_configs_online:
        sources = {}
        for name in ('tokenizer_config.json', 'generation_config.json'):
            url = 'https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/' + name
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read(1 << 20)
                sources[name] = {'url': url, 'sha256': hashlib.sha256(raw).hexdigest(),
                                 'repo_commit': response.headers.get('x-repo-commit')}
            if name == 'tokenizer_config.json':
                sources[name]['local_asset_equal'] = (
                    raw == (args.reference_tokenizer / name).read_bytes())
                if not sources[name]['local_asset_equal']:
                    raise AssertionError('Official tokenizer config changed; inspect before proceeding')
            else:
                sources[name]['configuration'] = json.loads(raw)
        result['official_sources'] = sources

    source = subprocess.check_output(
        ['git', 'show', args.baseline_ref + ':air_llm/airllm/nextgen/gguf_tokenizer.py'],
        cwd=ROOT, text=True)
    namespace = {}
    exec(compile(source, '<baseline tokenizer>', 'exec'), namespace)
    reference = AutoTokenizer.from_pretrained(args.reference_tokenizer, local_files_only=True)
    official_engine = tokenizers.Tokenizer.from_file(str(args.reference_tokenizer / 'tokenizer.json'))
    reader = GGUFReader(args.model, memory_budget_bytes=256 << 20, max_metadata_bytes=192 << 20)
    with reader:
        tokenizer = GGUFTokenizer(reader.metadata)
        baseline = namespace['GGUFTokenizer'](reader.metadata)
        official_json = json.loads((args.reference_tokenizer / 'tokenizer.json').read_text())
        official_merges = official_json['model']['merges']
        official_merges = [' '.join(m) if isinstance(m, list) else m for m in official_merges]
        result['gguf_official_assets'] = {
            'official_vocab_ids_equal': all(reader.metadata['tokenizer.ggml.tokens'][v] == k
                                           for k, v in official_json['model']['vocab'].items()),
            'all_merges_equal': reader.metadata['tokenizer.ggml.merges'] == official_merges,
            'merges_count': len(official_merges),
        }
        result['special_tokens'] = {
            name: {'gguf_id': reader.metadata['tokenizer.ggml.tokens'].index(name),
                   'official_id': reference.convert_tokens_to_ids(name)}
            for name in ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>')}
        result['eos'] = {'tokenizer_id': tokenizer.eos_token_id,
                         'official_tokenizer_id': reference.eos_token_id,
                         'gguf_padding_id': reader.metadata.get('tokenizer.ggml.padding_token_id')}
        texts = ['Été e\u0301lan 中文 مرحبا 123456 3.14159',
                 'Hello\n\n  world!\r\n', '\t début\u00a0 fin  ', '',
                 '<think>abc</think><tool_call>{"a":1}</tool_call>',
                 '👩🏽\u200d💻 café\u0301 naïve क् हिन्दी',
                 '<|im_start|>user\nBonjour<|im_end|>\n',
                 '2 + 2 = 4.\n0.000001 1234567890',
                 '<|endoftext|><|im_end|><think>\n\n</think>\n\n']
        result['text_cases'] = []
        for text in texts:
            own = tokenizer.encode(text)
            ref = official_engine.encode(text, add_special_tokens=False).ids
            hf_ids = reference.encode(text, add_special_tokens=False)
            entry = {'text': text, 'ids': own, 'official_ids': ref, 'ids_equal': own == ref,
                     'transformers_ids': hf_ids, 'transformers_ids_equal': own == hf_ids}
            for skip in (False, True):
                own_decode = tokenizer.decode(own, skip_special_tokens=skip)
                ref_decode = official_engine.decode(ref, skip_special_tokens=skip)
                entry['decode_skip_' + str(skip).lower()] = {
                    'own': own_decode, 'official': ref_decode, 'equal': own_decode == ref_decode}
            result['text_cases'].append(entry)

        cases = [
            ('single_user', [{'role': 'user', 'content': 'Bonjour'}]),
            ('system_user', [{'role': 'system', 'content': 'Réponds en français.'},
                             {'role': 'user', 'content': 'Combien font 2 + 2 ?'}]),
            ('multi_turn', [{'role': 'system', 'content': 'Sois précis.'},
                            {'role': 'user', 'content': 'Retient le code 7391.'},
                            {'role': 'assistant', 'content': 'Le code est 7391.'},
                            {'role': 'user', 'content': 'Quel est le code ?'}]),
            ('whitespace', [{'role': 'system', 'content': '\n  Français. \n'},
                            {'role': 'user', 'content': '\t  Éte\u0301\n  suite  \r\n'},
                            {'role': 'assistant', 'content': '\n  Réponse  \n'},
                            {'role': 'user', 'content': '  Après\n'}]),
            ('completed_assistant', [{'role': 'user', 'content': 'A'},
                                     {'role': 'assistant', 'content': 'B'}]),
            ('embedded_thinking', [{'role': 'user', 'content': 'A'},
                                  {'role': 'assistant', 'content': '<think>\n  Calcul  \n</think>\n\nRéponse'},
                                  {'role': 'user', 'content': 'Suite'}]),
            ('reasoning_content', [{'role': 'user', 'content': 'A'},
                                  {'role': 'assistant', 'content': 'Réponse', 'reasoning_content': '  Calcul  '},
                                  {'role': 'user', 'content': 'Suite'}]),
            ('empty_system_and_answer', [{'role': 'system', 'content': '  '},
                                         {'role': 'user', 'content': 'A'},
                                         {'role': 'assistant', 'content': ''}]),
        ]
        result['chat_cases'] = []
        for (name, messages), thinking, preserve, generation in itertools.product(cases, (False, True), (False, True), (False, True)):
            kwargs = dict(enable_thinking=thinking, preserve_thinking=preserve, add_generation_prompt=generation)
            expected_text = reference.apply_chat_template(messages, tokenize=False, **kwargs)
            expected_ids = official_engine.encode(expected_text, add_special_tokens=False).ids
            actual_text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
            actual_ids = tokenizer.apply_chat_template(messages, **kwargs)
            entry = {'name': name, 'messages': messages, 'options': kwargs,
                     'rendered_text': actual_text, 'official_text': expected_text,
                     'text_equal': actual_text == expected_text,
                     'ids': actual_ids, 'official_ids': list(expected_ids),
                     'ids_equal': actual_ids == list(expected_ids)}
            old_kwargs = {k: v for k, v in kwargs.items() if k != 'preserve_thinking'}
            # The baseline has no preserve_thinking option. Compare only its
            # supported default, rather than claim failures for an absent API.
            if not preserve:
                try:
                    old_text = baseline.apply_chat_template(messages, tokenize=False, **old_kwargs)
                    entry['baseline_text_equal'] = old_text == expected_text
                    if old_text != expected_text:
                        entry['baseline_text'] = old_text
                except Exception as error:
                    entry['baseline_error'] = type(error).__name__ + ': ' + str(error)
                    entry['baseline_text_equal'] = False
            result['chat_cases'].append(entry)

        bad_cases = [
            [], [{'role': 'system', 'content': 'A'}],
            [{'role': 'assistant', 'content': 'A'}],
            [{'role': 'user', 'content': 'A'}, {'role': 'system', 'content': 'B'}],
            [{'role': 'user', 'content': '<tool_response>A</tool_response>'}],
        ]
        result['invalid_conversations'] = []
        for messages in bad_cases:
            entry = {'messages': messages}
            for name, engine in [('own', tokenizer), ('official', reference), ('baseline', baseline)]:
                try:
                    engine.apply_chat_template(messages, tokenize=False, enable_thinking=False)
                    entry[name + '_rejected'] = False
                except Exception as error:
                    entry[name + '_rejected'] = True
                    entry[name + '_error'] = type(error).__name__ + ': ' + str(error)
            result['invalid_conversations'].append(entry)
        result['reader_stats'] = dict(reader.stats)

    result['summary'] = {
        'text_cases': len(result['text_cases']),
        'text_tokenization_equal': all(c['ids_equal'] for c in result['text_cases']),
        'text_decoding_equal': all(c['decode_skip_' + str(skip).lower()]['equal']
                                   for c in result['text_cases'] for skip in (False, True)),
        'transformers_text_tokenization_equal': all(c['transformers_ids_equal'] for c in result['text_cases']),
        'chat_cases': len(result['chat_cases']),
        'chat_text_and_ids_equal': all(c['text_equal'] and c['ids_equal'] for c in result['chat_cases']),
        'baseline_default_cases': sum('baseline_text_equal' in c for c in result['chat_cases']),
        'baseline_default_cases_failed': sum(c.get('baseline_text_equal') is False for c in result['chat_cases']),
        'invalid_conversations_rejected': all(c['own_rejected'] and c['official_rejected'] for c in result['invalid_conversations']),
        'tensor_bytes_read': result['reader_stats']['tensor_bytes'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result['summary'], ensure_ascii=False))
    if not all(result['summary'][name] for name in (
            'text_tokenization_equal', 'text_decoding_equal', 'chat_text_and_ids_equal', 'invalid_conversations_rejected')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
