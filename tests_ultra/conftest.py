from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'air_llm'))
torch.set_num_threads(1)


def save_tiny(root, kind='llama', *, tied=False):
    from transformers import (LlamaConfig, LlamaForCausalLM, MixtralConfig, MixtralForCausalLM,
                              GlmMoeDsaConfig, GlmMoeDsaForCausalLM, PreTrainedTokenizerFast)
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    torch.manual_seed(521)
    common = dict(vocab_size=96, hidden_size=48, intermediate_size=96,
                  num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=128, pad_token_id=0, bos_token_id=1,
                  eos_token_id=2, tie_word_embeddings=tied)
    if kind == 'mixtral':
        cfg = MixtralConfig(**common, num_local_experts=8, num_experts_per_tok=2)
        model = MixtralForCausalLM(cfg)
    elif kind == 'glm':
        common['num_key_value_heads'] = common['num_attention_heads']
        cfg = GlmMoeDsaConfig(**common, n_routed_experts=8, num_experts_per_tok=2,
                             n_shared_experts=1, first_k_dense_replace=1,
                             moe_intermediate_size=24, kv_lora_rank=8, q_lora_rank=16,
                             qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
                             index_head_dim=8, index_n_heads=2, index_topk=4,
                             indexer_types=['full', 'shared', 'full'],
                             rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0})
        model = GlmMoeDsaForCausalLM(cfg)
    else:
        cfg = LlamaConfig(**common, attention_bias=False, mlp_bias=False)
        model = LlamaForCausalLM(cfg)
    model.eval()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(root, safe_serialization=True, max_shard_size='100KB')
    tokens = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3}
    tokens.update({f't{i}': i for i in range(4, 96)})
    tokenizer = Tokenizer(WordLevel(tokens, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', bos_token='[BOS]',
                           eos_token='[EOS]', unk_token='[UNK]').save_pretrained(root)
    return model


@pytest.fixture(params=['llama', 'mixtral', 'glm'])
def tiny(request, tmp_path):
    root = tmp_path / request.param
    model = save_tiny(root, request.param)
    return root, model
