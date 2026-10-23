"""Pinned auxiliary file identities and interrupted range-resume behavior."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('granite_download',
    Path(__file__).resolve().parents[1]/'benchmarks/download_granite_checkpoint.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_non_lfs_file_matches_git_and_rejects_same_size_corruption(tmp_path):
    file = tmp_path/'config.json'
    file.write_text('{"a":1}\n')
    git_sha = subprocess.run(['git','hash-object',str(file)],capture_output=True,text=True,check=True).stdout.strip()
    entry = SimpleNamespace(rfilename=file.name,size=file.stat().st_size,lfs=None,blob_id=git_sha)
    assert download.verify_file(file,entry)['git_blob_sha1'] == git_sha
    file.write_text('{"a":2}\n')
    with pytest.raises(ValueError,match='content identity'):
        download.verify_file(file,entry)


def test_lfs_file_requires_sha256_even_with_same_size(tmp_path):
    file = tmp_path/'model.safetensors'
    file.write_bytes(b'checkpoint bytes')
    entry = SimpleNamespace(rfilename=file.name,size=file.stat().st_size,
        lfs={'sha256':hashlib.sha256(b'checkpoint bytes').hexdigest()},blob_id='pointer-blob')
    assert download.verify_file(file,entry)['hub_lfs_sha256'] == entry.lfs['sha256']
    file.write_bytes(b'corruption bytes')
    with pytest.raises(ValueError,match='content identity'):
        download.verify_file(file,entry)


def test_range_resume_recovers_from_partial_markers_and_rechecks_payload(tmp_path):
    file, marker = tmp_path/'payload',tmp_path/'0-3'
    file.write_bytes(b'abcdwxyz')
    fd = os.open(file,os.O_RDONLY)
    try:
        assert not download.reusable_range(fd,marker,0,3)
        for broken in ('{"sha256":', 'null', '[]'):
            marker.write_text(broken)
            assert not download.reusable_range(fd,marker,0,3)
        values = {'start':0,'end':3,'sha256':hashlib.sha256(b'abcd').hexdigest()}
        download.write_marker(marker,values)
        assert json.loads(marker.read_text()) == values
        assert download.reusable_range(fd,marker,0,3)
        assert not list(tmp_path.glob('*.tmp-*'))
        assert not download.reusable_range(fd,marker,4,7)
        file.write_bytes(b'zzzzwxyz')
        assert not download.reusable_range(fd,marker,0,3)
    finally:
        os.close(fd)
