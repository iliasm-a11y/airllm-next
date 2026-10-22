"""
bench_m5_io.py — bancs de lecture stockage et mémoire sur Apple Silicon (macOS).

Usage :
  python bench_m5_io.py --file /chemin/test.bin --size-gb 8 --create   # SSD interne (crée puis supprime)
  python bench_m5_io.py --file /Volumes/SSD/model-00001.safetensors        # fichier existant, lecture seule

Mesures (F_NOCACHE demandé sur chaque descripteur) :
  - lecture séquentielle par blocs (1, 4, 16, 37.75, 64 Mo) en mono-thread ;
  - lectures d'experts à offsets aléatoires (taille paramétrable) avec 1/2/4/8 threads (os.pread libère le GIL) ;
  - bande passante de copie mémoire (numpy).
Aucune écriture sur un fichier existant ; --create écrit un fichier neuf avec F_NOCACHE puis le supprime.
"""
import os, sys, time, json, fcntl, random, argparse, threading, platform
import numpy as np

F_NOCACHE = 48  # <sys/fcntl.h> Darwin

def open_nocache(path, flags=os.O_RDONLY):
    fd = os.open(path, flags)
    ok = True
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    except OSError:
        ok = False
    return fd, ok

def create_file(path, size_gb):
    chunk = 64 << 20
    rng = np.random.default_rng(0)
    buf = rng.integers(0, 256, size=chunk, dtype=np.uint8).tobytes()
    fd, ok = open_nocache(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    n = int(size_gb * (1 << 30) // chunk)
    t = time.perf_counter()
    for i in range(n):
        # varier le contenu pour éviter toute déduplication/compression transparente
        os.write(fd, buf[i % 7:] + buf[: i % 7])
    os.fsync(fd); os.close(fd)
    dt = time.perf_counter() - t
    return dict(bytes=n * chunk, seconds=dt, write_GBps=n * chunk / dt / 1e9, nocache=ok)

def seq_read(path, block, max_bytes):
    fd, ok = open_nocache(path)
    size = os.fstat(fd).st_size
    total = min(size, max_bytes)
    buf = bytearray(block)
    mv = memoryview(buf)
    off = 0; t = time.perf_counter(); got = 0
    while off + block <= total:
        n = os.preadv(fd, [mv], off) if hasattr(os, 'preadv') else len(os.pread(fd, block, off))
        got += n; off += block
    dt = time.perf_counter() - t
    os.close(fd)
    return dict(block_MB=block / 1e6, bytes=got, seconds=dt, GBps=got / dt / 1e9, nocache=ok)

def rand_read(path, block, n_reads, threads, seed=1):
    fd, ok = open_nocache(path)
    size = os.fstat(fd).st_size
    rng = random.Random(seed)
    align = 16384
    offs = [rng.randrange(0, (size - block) // align) * align for _ in range(n_reads)]
    lock = threading.Lock(); idx = [0]; got = [0]
    def worker():
        buf = bytearray(block); mv = memoryview(buf)
        while True:
            with lock:
                if idx[0] >= len(offs): return
                o = offs[idx[0]]; idx[0] += 1
            n = os.preadv(fd, [mv], o) if hasattr(os, 'preadv') else len(os.pread(fd, block, o))
            with lock: got[0] += n
    t = time.perf_counter()
    ths = [threading.Thread(target=worker) for _ in range(threads)]
    [th.start() for th in ths]; [th.join() for th in ths]
    dt = time.perf_counter() - t
    os.close(fd)
    return dict(block_MB=block / 1e6, threads=threads, reads=n_reads, bytes=got[0], seconds=dt, GBps=got[0] / dt / 1e9, nocache=ok)

def mem_bw(size_mb=1024, reps=5):
    a = np.ones(size_mb << 20, dtype=np.uint8); b = np.empty_like(a)
    np.copyto(b, a)
    t = time.perf_counter()
    for _ in range(reps): np.copyto(b, a)
    dt = time.perf_counter() - t
    return dict(copy_GBps_read_plus_write=2 * reps * a.nbytes / dt / 1e9, single_thread=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True); ap.add_argument('--size-gb', type=float, default=8)
    ap.add_argument('--create', action='store_true'); ap.add_argument('--expert-mb', type=float, default=37.76)
    ap.add_argument('--out', default='bench_io.json'); ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args()
    res = dict(platform=platform.platform(), machine=platform.machine(), started=time.time(), runs=[])
    if a.create:
        res['create'] = create_file(a.file, a.size_gb)
    size = os.path.getsize(a.file); maxb = min(size, 6 << 30)
    expert = int(a.expert_mb * 1e6) // 16384 * 16384
    plan = []
    for rep in range(a.reps):
        for blk in [1 << 20, 4 << 20, 16 << 20, expert, 64 << 20]:
            plan.append(('seq', blk, None))
        for th in [1, 2, 4, 8]:
            plan.append(('rand', expert, th))
            plan.append(('rand', 4 << 20, th))
    random.Random(42).shuffle(plan)
    for kind, blk, th in plan:
        if kind == 'seq':
            r = seq_read(a.file, blk, maxb); r['kind'] = 'seq'
        else:
            n = max(8, int((3 << 30) // blk))
            r = rand_read(a.file, blk, n, th, seed=random.randrange(10**6)); r['kind'] = 'rand'
        res['runs'].append(r); print(json.dumps(r), flush=True)
    res['mem'] = mem_bw()
    res['loadavg_end'] = os.getloadavg()
    json.dump(res, open(a.out, 'w'), indent=1)
    if a.create:
        os.remove(a.file)
        res['deleted_test_file'] = True
        json.dump(res, open(a.out, 'w'), indent=1)
    print('saved', a.out)
