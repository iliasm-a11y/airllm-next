"""Preparation boundary checks only; no checkpoint payload/native command."""
from pathlib import Path
from types import SimpleNamespace
import json
from research.glm53_gpu_v3 import dense_cold_warm


def test_prepare_does_not_read_checkpoint_payload_or_start_native_tool(tmp_path,monkeypatch):
    capture=tmp_path/'capture';capture.mkdir()
    (capture/'activation.bin').write_bytes(bytes(16384))
    (capture/'tensors.jsonl').write_text(json.dumps({'step':1,'name':'attn_norm-40','shape':[4096,1,1,1],
        'type':0,'bytes':16384,'strides':[4,16384,16384,16384],'file':'activation.bin'})+'\n')
    shard=tmp_path/'shard.gguf';shard.write_bytes(b'checkpoint payload must never be opened')
    inventory=tmp_path/'inventory.json'
    inventory.write_text(json.dumps({'revision':'fixture','tensors':[{'name':'blk.0.attn_q.weight','type':'Q8_0',
        'dims':[4096,8192],'offset':64,'file':shard.name,'bytes_with_alignment':35651584}],
        'files':[{'file':shard.name,'data_offset':128}]}))
    real_open=Path.open
    def deny_checkpoint_open(path,*args,**kwargs):
        if path.resolve()==shard.resolve():raise AssertionError('preparation read the checkpoint payload')
        return real_open(path,*args,**kwargs)
    def deny_native(*args,**kwargs):raise AssertionError('preparation launched a native tool')
    monkeypatch.setattr(Path,'open',deny_checkpoint_open)
    monkeypatch.setattr(dense_cold_warm.subprocess,'run',deny_native)
    monkeypatch.setattr(dense_cold_warm.subprocess,'Popen',deny_native)
    args=SimpleNamespace(capture=capture,inventory=inventory,directory=tmp_path/'component',output=tmp_path/'report.json',
                         model_directory=tmp_path,execute=False)
    report=dense_cold_warm.run(args)
    assert report['status']=='prepared_no_checkpoint_payload_read'
    assert report['absolute_offset']==192 and report['known_payload_bound_bytes']<128*1024**2
    assert report['phases']==[] and not report['gpu_initialized'] and not report['model_decoder_executed']
