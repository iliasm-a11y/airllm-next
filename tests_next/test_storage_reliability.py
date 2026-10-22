"""Storage failure regressions that need neither MLX nor framework imports."""
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import struct
from threading import Event

import numpy as np
import pytest

from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.config import NextConfig
from airllm.nextgen.lossless import LosslessCheckpoint, LosslessOverlay, open_checkpoint, pack_checkpoint
from airllm.nextgen.sidecar import QuantizedSidecar
from airllm.ultra.checkpoint import DirectCheckpoint


def checkpoint(tmp_path, *, limit=4096):
    root=tmp_path/'source'
    root.mkdir()
    arrays={'small':np.arange(64,dtype='<f4').reshape(4,16),
            'large':np.arange(256,dtype='<f4').reshape(16,16)}
    header={};payload=bytearray()
    for key,value in arrays.items():
        start=len(payload);payload.extend(value.tobytes())
        header[key]={'dtype':'F32','shape':list(value.shape),
                     'data_offsets':[start,len(payload)]}
    raw=json.dumps(header).encode();raw+=b' '*(-len(raw)%8)
    (root/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+payload)
    return OffsetCheckpoint(root,max_read_bytes=limit),arrays


def test_offset_close_disallows_reopening_during_parent_cleanup(tmp_path,monkeypatch):
    cp,_=checkpoint(tmp_path)
    cp.read_array('small')
    original=DirectCheckpoint.close
    def finish(reader):
        # This phase used to run after io_lock was released with closed=False.
        with pytest.raises(RuntimeError,match='closed'):
            with reader.descriptor('model.safetensors'):
                pass
        original(reader)
    monkeypatch.setattr(DirectCheckpoint,'close',finish)
    cp.close()
    assert not cp.fds
    cp.close()


@pytest.mark.parametrize('rows',[(0.,1),(False,1),(0,True)])
def test_invalid_selectors_fail_before_opening_data(tmp_path,rows):
    with checkpoint(tmp_path)[0] as cp:
        with pytest.raises(ValueError,match='row range'):
            cp.read_array('small',rows=rows)
        assert cp.stats['fd_opens']==0


def test_lossless_generator_batch_keeps_all_fallback_requests(tmp_path):
    cp,arrays=checkpoint(tmp_path)
    pack_checkpoint(cp,tmp_path/'pack',keys=['small'],block_bytes=128)
    reader=LosslessOverlay(cp,tmp_path/'pack')
    try:
        got=reader.read_arrays(key for key in ('large','large'))
        assert len(got)==2
        for value in got:np.testing.assert_array_equal(value,arrays['large'])
    finally:reader.close()


def test_lossless_close_preserves_inflight_descriptor_borrows(tmp_path,monkeypatch):
    import airllm.nextgen.lossless as lossless
    cp,arrays=checkpoint(tmp_path)
    pack_checkpoint(cp,tmp_path/'pack',keys=['small'],block_bytes=128)
    reader=LosslessOverlay(cp,tmp_path/'pack')
    started=Event();release=Event();original=lossless._read_exact
    def paused_read(*args):
        if not started.is_set():
            started.set()
            assert release.wait(3)
        return original(*args)
    monkeypatch.setattr(lossless,'_read_exact',paused_read)
    with ThreadPoolExecutor(1) as pool:
        future=pool.submit(reader.read_array,'small')
        try:
            assert started.wait(3)
            reader.close()
        finally:release.set()
        np.testing.assert_array_equal(future.result(timeout=3),arrays['small'])
    with pytest.raises(RuntimeError,match='closed'):reader.read_array('small')


def test_lossless_mixed_batch_preflights_every_selection(tmp_path):
    cp,_=checkpoint(tmp_path,limit=512)
    pack_checkpoint(cp,tmp_path/'pack',keys=['small'],block_bytes=128)
    reader=LosslessOverlay(cp,tmp_path/'pack')
    try:
        with pytest.raises(MemoryError):
            reader.read_arrays(['small','large'],max_batch_bytes=4096)
        assert reader.transport_stats['payload_bytes']==0
        assert cp.stats['requested_bytes']==0
    finally:reader.close()


def test_sidecar_manifest_failure_can_retry_on_same_object(tmp_path,monkeypatch):
    store=QuantizedSidecar(tmp_path/'sidecar','source',create=True)
    w=np.ones((4,64),dtype='f4')
    original=store.commit
    def failed_commit():raise OSError('manifest disk full')
    monkeypatch.setattr(store,'commit',failed_commit)
    with pytest.raises(OSError,match='disk full'):store.add('expert',[w],w.shape)
    assert not store.entries
    assert json.loads(store.path.read_text())['entries']=={}
    monkeypatch.setattr(store,'commit',original)
    assert store.add('expert',[w],w.shape)
    resumed=QuantizedSidecar(store.root,'source')
    assert resumed.verify()==['expert']


def test_failed_lossless_manifest_commit_never_exposes_pack(tmp_path,monkeypatch):
    import airllm.nextgen.lossless as lossless
    cp,_=checkpoint(tmp_path)
    def failed_commit(*args):raise OSError('manifest disk full')
    monkeypatch.setattr(lossless,'atomic_json',failed_commit)
    try:
        with pytest.raises(OSError):pack_checkpoint(cp,tmp_path/'pack',keys=['small'],block_bytes=128)
        assert not (tmp_path/'pack/manifest.json').exists()
        with pytest.raises(FileNotFoundError):LosslessOverlay(cp,tmp_path/'pack')
    finally:cp.close()


@pytest.mark.parametrize('codec',['raw','zlib','shuffle-zlib','palette16','auto'])
def test_standalone_pack_reads_exact_bytes_without_original_shards(tmp_path,codec):
    cp,arrays=checkpoint(tmp_path)
    (cp.root/'config.json').write_text('{"model_type":"fixture"}')
    (cp.root/'tokenizer.json').write_text('{"fixture":"tokenizer asset"}')
    (cp.root/'modeling_custom.py').write_text('raise RuntimeError("must never execute")')
    destination=tmp_path/'pack'
    meta=pack_checkpoint(cp,destination,block_bytes=128,codec=codec,standalone=True)
    assert set(meta['assets'])=={'config.json','tokenizer.json'}
    assert not (destination/'modeling_custom.py').exists()
    cp.close();shutil.rmtree(cp.root)
    cfg=NextConfig(max_tensor_bytes=4096,workspace_bytes=4096,io_workers=1)
    with open_checkpoint(destination,cfg) as reader:
        assert isinstance(reader,LosslessCheckpoint)
        for key,value in arrays.items():
            assert reader.read_array(key).tobytes()==value.tobytes()
            assert reader.read_array(key,rows=(1,3)).tobytes()==value[1:3].tobytes()
        assert reader.total_bytes==sum(v.nbytes for v in arrays.values())
        assert reader.stats['standalone_pack']==1
        assert reader.read_arrays(iter(()))==[]


def test_standalone_refuses_partial_pack_and_oversize_assets_before_writing(tmp_path):
    cp,_=checkpoint(tmp_path)
    (cp.root/'config.json').write_text('{"fixture":true}')
    try:
        with pytest.raises(ValueError,match='every source tensor'):
            pack_checkpoint(cp,tmp_path/'partial',keys=['small'],standalone=True)
        assert not (tmp_path/'partial').exists()
        with pytest.raises(MemoryError,match='max_asset_bytes'):
            pack_checkpoint(cp,tmp_path/'oversize',standalone=True,max_asset_bytes=1)
        assert not (tmp_path/'oversize').exists()
    finally:cp.close()


def test_standalone_asset_and_layout_corruption_fail_closed(tmp_path):
    cp,_=checkpoint(tmp_path)
    config='{"fixture":true}'
    (cp.root/'config.json').write_text(config)
    destination=tmp_path/'pack'
    pack_checkpoint(cp,destination,standalone=True,block_bytes=128)
    cp.close()
    (destination/'config.json').write_text(config.replace('true','null'))
    with pytest.raises(ValueError,match='asset checksum'):LosslessCheckpoint(destination)
    (destination/'config.json').write_text(config)
    manifest=destination/'manifest.json';meta=json.loads(manifest.read_text())
    meta['entries']['small']['raw_size']+=1;manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='byte count'):LosslessCheckpoint(destination)


def test_standalone_asset_symlink_cannot_escape_source(tmp_path):
    cp,_=checkpoint(tmp_path)
    external=tmp_path/'external.json';external.write_text('{}')
    (cp.root/'config.json').symlink_to(external)
    try:
        with pytest.raises(ValueError,match='escapes'):
            pack_checkpoint(cp,tmp_path/'pack',standalone=True)
        assert not (tmp_path/'pack').exists()
    finally:cp.close()
