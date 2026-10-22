"""
bench_m5_compute.py — coût de calcul d'un expert MoE et de la déquantification FP8, CPU (torch).

Mesure :
  1. temps d'un FFN SwiGLU d'expert (GLM : 6144x2048 ; Qwen3.8 : 8192x2048) en FP32 selon le nombre
     de tokens n = 1..8192 (plat tant que la lecture des poids domine) ;
  2. débit de déquantification FP8 e4m3 -> FP32 / BF16 (torch natif et table de correspondance numpy),
     avec mise à l'échelle par blocs 128x128 ;
Le pendant GPU (Metal, MLX) est dans bench_m5_metal.py, à exécuter hors bac à sable.
"""
import time, json, os, sys, numpy as np, torch

def t_expert(H, I, n, reps=3, dtype=torch.float32):
    wg = torch.randn(I, H, dtype=dtype) * 0.02; wu = torch.randn(I, H, dtype=dtype) * 0.02; wd = torch.randn(H, I, dtype=dtype) * 0.02
    x = torch.randn(n, H, dtype=dtype)
    def f():
        h = torch.nn.functional.silu(x @ wg.T) * (x @ wu.T)
        return h @ wd.T
    f()
    t = time.perf_counter()
    for _ in range(reps): f()
    dt = (time.perf_counter() - t) / reps
    return dict(H=H, I=I, n=n, seconds=dt, GFLOPS=6 * H * I * n / dt / 1e9,
                weight_MB_fp8=3 * H * I / 1e6)

def dequant_fp8(rows=2048, cols=6144, reps=5):
    q = torch.randint(0, 256, (rows, cols), dtype=torch.uint8)
    q[(q & 0x7F) == 0x7F] = 0  # éviter NaN e4m3fn
    s = torch.rand(rows // 128, cols // 128, dtype=torch.float32) + 0.5
    qf8 = q.view(torch.float8_e4m3fn)
    out = {}
    for name, dt in [('fp32', torch.float32), ('bf16', torch.bfloat16)]:
        def f():
            w = qf8.to(torch.float32).view(rows // 128, 128, cols // 128, 128) * s[:, None, :, None]
            return w.reshape(rows, cols).to(dt)
        f(); t = time.perf_counter()
        for _ in range(reps): f()
        dtm = (time.perf_counter() - t) / reps
        out['torch_' + name + '_GB_fp8_per_s'] = rows * cols / dtm / 1e9
    # LUT numpy (mono-thread)
    import ml_dtypes
    lut = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    lut[np.isnan(lut)] = 0
    qn = q.numpy()
    t = time.perf_counter()
    for _ in range(reps): w = lut[qn]
    out['numpy_lut_fp32_GB_fp8_per_s'] = rows * cols * reps / (time.perf_counter() - t) / 1e9
    return out

if __name__ == '__main__':
    res = dict(torch=torch.__version__, threads=torch.get_num_threads(), loadavg_start=os.getloadavg(), expert=[], dequant=None)
    for (H, I) in [(6144, 2048), (8192, 2048)]:
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
            r = t_expert(H, I, n, reps=3 if n < 2048 else 1)
            res['expert'].append(r); print(json.dumps(r), flush=True)
    res['dequant'] = dequant_fp8(); print(res['dequant'])
    res['loadavg_end'] = os.getloadavg()
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else 'bench_compute.json', 'w'), indent=1)
