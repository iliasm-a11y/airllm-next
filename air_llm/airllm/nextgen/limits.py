"""Optimistic hardware bounds. These are formulas, never observed throughput."""
import math


def estimate_glm(config,cache_bytes=2*1024**3,ssd_gbps=(.5,1.,2.,3.)):
    c=config;H=c['hidden_size'];M=c['moe_intermediate_size'];E=c['n_routed_experts'];K=c['num_experts_per_tok']
    L=sum(x=='sparse' for x in c.get('mlp_layer_types',[])) or c['num_hidden_layers']-c.get('first_k_dense_replace',1)
    active=L*K*3*H*M;total=L*E*3*H*M
    br,bc=c.get('quantization_config',{}).get('weight_block_size',[128,128])
    scales_per_expert=2*math.ceil(M/br)*math.ceil(H/bc)+math.ceil(H/br)*math.ceil(M/bc)
    fp8=active+L*K*scales_per_expert*4
    formats={'official_fp8':{'active_bytes':fp8,'expert_storage_bytes':total+L*E*scales_per_expert*4,'resident_bytes_per_active_parameter':2},
      'affine_int8_g64_f32_scales':{'active_bytes':active*(1+8/64),'expert_storage_bytes':total*(1+8/64),'resident_bytes_per_active_parameter':1+8/64},
      'affine_int4_g64_f32_scales':{'active_bytes':active*(.5+8/64),'expert_storage_bytes':total*(.5+8/64),'resident_bytes_per_active_parameter':.5+8/64}}
    A=c['num_attention_heads'];Q=c['q_lora_rank'];R=c['kv_lora_rank'];N=c['qk_nope_head_dim'];D=c['qk_rope_head_dim'];V=c['v_head_dim'];layers=c['num_hidden_layers']
    full=sum(x=='full' for x in c['indexer_types'])
    attn=layers*(H*Q+Q*A*(N+D)+H*(R+D)+R*A*(N+V)+A*V*H)
    kv_b=layers*R*A*(N+V)
    dense=(layers-L)*3*H*c['intermediate_size'];shared=L*3*H*M
    index=full*(Q*c['index_n_heads']*c['index_head_dim']+H*c['index_head_dim']+H*c['index_n_heads'])
    router=L*E*H;head=H*c['vocab_size']
    # Approximation from architecture/config, not measured checkpoint headers.
    # Normalization/scales and metadata overhead are omitted here.
    fixed_original=attn+dense+shared+index+2*router+2*head+2*H
    linear_convertible=attn-kv_b+dense+shared+head
    retained=kv_b+index+2*router+2*H
    full_decode_scenarios={
      'fp8_original_approx_bytes_per_token':fp8+fixed_original,
      'experts_int4_only_approx_bytes_per_token':active*.625+fixed_original,
      'mixed_int4_approx_bytes_per_token':active*.625+linear_convertible*.625+retained,
      'mixed_expert4_linear8_approx_bytes_per_token':active*.625+linear_convertible*1.125+retained}
    bounds=[]
    for name,v in formats.items():
        disk_per_parameter=v['active_bytes']/active
        # An extremely optimistic fully useful cache: all resident bytes save
        # active weights every token. Real hot sets and non-expert weights differ.
        saved_parameters=min(active,cache_bytes/v['resident_bytes_per_active_parameter'])
        minimum=max(0,v['active_bytes']-saved_parameters*disk_per_parameter)
        for bandwidth in ssd_gbps:
            bounds.append({'format':name,'ssd_sustained_GB_s_assumed':bandwidth,'expert_only_active_GB':v['active_bytes']/1e9,
              'optimistic_expert_GB_after_cache':minimum/1e9,'optimistic_tokens_s_io_bound':bandwidth*1e9/minimum if minimum else None,
              'optimistic_seconds_per_token':minimum/(bandwidth*1e9)})
    # KV capacity is the actual latent representation, not expanded all-head KV.
    full=sum(x=='full' for x in c['indexer_types']);kv_per_token=2*(c['num_hidden_layers']*(c['kv_lora_rank']+c['qk_rope_head_dim'])+full*c['index_head_dim'])
    return {'kind':'theoretical lower time bounds; not benchmarks','active_routed_expert_parameters_per_token':active,
      'total_routed_expert_parameters':total,'formats':formats,'ssd_bounds':bounds,
      'full_decode_no_cache_estimates':full_decode_scenarios,
      'full_decode_estimate_assumptions':'architecture-derived: major linear weights FP8, router/head BF16; normalization/scale overhead excluded; actual headers and cache locality must be measured',
      'latent_kv_bytes_per_token_bf16':kv_per_token,'latent_kv_1M_tokens_GB':kv_per_token*1048576/1e9,
      'exclusions':['attention/shared/dense/output weights IO','dequantization','compute','OS/filesystem overhead','thermal throttling','imperfect cache reuse','Python/GPU synchronizations'],
      'apple_M5_memory_bandwidth_GB_s_specification':153,'memory_bandwidth_is_not_external_USB_throughput':True}
