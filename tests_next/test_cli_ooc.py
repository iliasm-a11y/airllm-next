import ast
import json
import numpy as np
import pytest
import torch
from airllm.nextgen.cli import main
from airllm.nextgen.sidecar import QuantizedSidecar
from airllm.nextgen.storage import OffsetCheckpoint


def test_cli_local_draft_performs_real_target_verification(tiny_glm, tmp_path, capsys):
    root, reference = tiny_glm
    report = tmp_path/'generation.json'
    ids = [[1, 8, 12]]
    with torch.inference_mode():
        expected = reference.generate(torch.tensor(ids), max_new_tokens=4, do_sample=False).tolist()[0]
    main(['run', str(root), '--device', 'cpu', '--float32', '--token-ids', '1,8,12',
          '--max-new-tokens', '4', '--draft-model', str(root), '--cache-mib', '1',
          '--io-mib', '1', '--kv-mib', '1', '--cache-dir', str(tmp_path/'metadata'),
          '--json-report', str(report)])
    result = ast.literal_eval(capsys.readouterr().out.strip())
    assert result == expected
    metrics = json.loads(report.read_text())
    assert metrics['runtime']['speculative']['mode'] == 'greedy_block_verification'
    assert metrics['runtime']['tokens_generated'] == len(result)-len(ids[0])


def test_cli_verify_audits_content_and_detects_corruption(tiny_glm, tmp_path, capsys):
    root, _ = tiny_glm
    quant = tmp_path/'packed'
    with OffsetCheckpoint(root) as checkpoint:
        store = QuantizedSidecar(quant, checkpoint.fingerprint, create=True)
    store.add('expert', [np.ones((16,64), dtype='f4')], (16,64))
    command = ['verify', str(root), '--quantized-dir', str(quant), '--cache-dir', str(tmp_path/'metadata')]
    main(command)
    report = json.loads(capsys.readouterr().out)
    assert report['verified_files'] == 1
    assert report['integrity_bytes'] > 0
    with (quant/store.entries['expert']['file']).open('r+b') as stream:
        stream.seek(-1,2)
        stream.write(b'\xff')
    with pytest.raises(ValueError,match='checksum mismatch'):
        main(command)
