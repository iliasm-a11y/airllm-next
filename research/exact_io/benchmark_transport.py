"""Compare the actual OffsetCheckpoint with its optional lossless overlay.

Warm OS cache, local internal disk only. This is a transport benchmark, not a
decode benchmark or a certified external SSD measurement.
"""
import argparse
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from common import ROOT, save
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.lossless import LosslessOverlay, NativePalette, pack_checkpoint


def run(repeats, native_library):
    source = ROOT/'samples'
    pack = ROOT/'build/sample-pack'
    cp = OffsetCheckpoint(source)
    if not pack.exists():
        pack_checkpoint(cp,pack,codec='palette16',block_bytes=256*1024)
    readers = {'offset_reference':cp,
        'palette_numpy_hash':LosslessOverlay(cp,pack),
        'palette_c_hash':LosslessOverlay(cp,pack,palette_decoder=NativePalette(native_library)),
        'palette_c_nohash':LosslessOverlay(cp,pack,palette_decoder=NativePalette(native_library),verify_hash=False)}
    keys = list(cp.tensors)
    # Compare every bit once outside timings; no allclose or numerical casting.
    for key in keys:
        expected = cp.read_array(key).tobytes()
        for reader in list(readers.values())[1:]:
            assert reader.read_array(key).tobytes() == expected
    trials = []
    rng = random.Random(20261022)
    work = [(name,r) for r in range(repeats) for name in readers]
    rng.shuffle(work)
    sink = 0
    for name,r in work:
        reader = readers[name]
        if name!='offset_reference':
            before = dict(reader.transport_stats)
        else: before = dict(cp.stats)
        t = time.perf_counter()
        for key in keys:
            a = reader.read_array(key)
            sink ^= int(a[0,0])
        elapsed = time.perf_counter()-t
        if name!='offset_reference':
            b = reader.transport_stats['payload_bytes']-before.get('payload_bytes',0)
            ix = reader.transport_stats['index_bytes']-before.get('index_bytes',0)
        else:
            b = cp.stats['logical_bytes']-before.get('logical_bytes',0)
            ix = 0
        trials.append(dict(mode=name,repeat=r,seconds=elapsed,encoded_payload_bytes=b,
            index_bytes=ix,requested_raw_bytes=cp.total_bytes,decoded_GBps=cp.total_bytes/elapsed/1e9))
    summary = []
    ref = statistics.median(t['seconds'] for t in trials if t['mode']=='offset_reference')
    for name in readers:
        ts = [t for t in trials if t['mode']==name]
        med = statistics.median(t['seconds'] for t in ts)
        summary.append(dict(mode=name,median_seconds=med,transport_speedup_vs_offset=ref/med,
            encoded_ratio_including_binary_index=(ts[0]['encoded_payload_bytes']+ts[0]['index_bytes'])/cp.total_bytes,
            median_decoded_GBps=cp.total_bytes/med/1e9))
    save('transport.json',dict(evidence='measured actual AirLLM OffsetCheckpoint/overlay, warm OS cache',
        repeats=repeats,order_seed=20261022,trials=trials,summary=summary,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        exact_sample_bits=True,physical_device_read_bytes=None,external_ssd_measured=False,
        limitations=['SHA256 disabled only in explicitly named nohash ablation; defaults verify checksums.',
        'Original checkpoint reader does not hash weight payloads on every access.',
        'Warm cache traffic is not SSD traffic or token throughput.'],sink=sink))
    print(summary,flush=True)
    for name,reader in readers.items():
        if name!='offset_reference': reader.close()
    cp.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--repeats',type=int,default=7)
    ap.add_argument('--native-library',type=Path,default=ROOT/'build/palette.dylib')
    a=ap.parse_args()
    run(a.repeats,a.native_library)
