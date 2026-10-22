"""
bench_m5_metal.py — à exécuter NATIVEMENT sur le Mac (hors bac à sable), dans un environnement MLX :
    python -m pip install mlx && python bench_m5_metal.py > metal.json

Mesure sur le GPU M5 (Metal) :
  1. FFN SwiGLU d'expert GLM (6144x2048) et Qwen3.8 (8192x2048) selon n = 1..8192 tokens :
     a) poids BF16 ; b) poids FP8 e4m3 stockés en uint8 et déquantifiés par table (gather) + échelle de bloc ;
     c) poids affines 4 bits (mx.quantized_matmul, groupe 64) ; d) mxfp4 / nvfp4 / mxfp8 si la version de MLX les offre.
  2. Débit de copie mémoire GPU (lecture+écriture).
Objectif : déterminer n_max (positions par passe pour lesquelles le calcul reste inférieur au temps de lecture SSD)
et vérifier que la déquantification FP8 ne devient pas le goulot.
"""
import time, json, mlx.core as mx

def timeit(f, reps=5):
    mx.eval(f()); t = time.perf_counter()
    for _ in range(reps): mx.eval(f())
    return (time.perf_counter() - t) / reps

def run():
    out = dict(mlx=mx.__version__, device=str(mx.default_device()), results=[])
    for (H, I) in [(6144, 2048), (8192, 2048)]:
        wg = mx.random.normal((I, H)).astype(mx.bfloat16) * 0.02
        wu = mx.random.normal((I, H)).astype(mx.bfloat16) * 0.02
        wd = mx.random.normal((H, I)).astype(mx.bfloat16) * 0.02
        q8 = [mx.random.randint(0, 120, w.shape).astype(mx.uint8) for w in (wg, wu, wd)]
        lut = mx.random.normal((256,)).astype(mx.bfloat16)
        qs = [mx.quantize(w, group_size=64, bits=4) for w in (wg, wu, wd)]
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
            x = mx.random.normal((n, H)).astype(mx.bfloat16)
            f_bf16 = lambda: (mx.sigmoid(x @ wg.T) * (x @ wg.T) * (x @ wu.T)) @ wd.T
            def f_fp8():
                g, u, d = (lut[q] for q in q8)
                return (mx.sigmoid(x @ g.T) * (x @ g.T) * (x @ u.T)) @ d.T
            def f_q4():
                a = mx.quantized_matmul(x, *qs[0], transpose=True, group_size=64, bits=4)
                b = mx.quantized_matmul(x, *qs[1], transpose=True, group_size=64, bits=4)
                h = mx.sigmoid(a) * a * b
                return mx.quantized_matmul(h, *qs[2], transpose=True, group_size=64, bits=4)
            row = dict(H=H, I=I, n=n)
            for name, f in [('bf16', f_bf16), ('fp8_lut', f_fp8), ('affine4', f_q4)]:
                dt = timeit(f, reps=5 if n <= 1024 else 2)
                row[name + '_s'] = dt; row[name + '_TFLOPS'] = 6 * H * I * n / dt / 1e12
            for mode in ['mxfp4', 'nvfp4', 'mxfp8']:
                try:
                    qm = [mx.quantize(w, mode=mode) for w in (wg, wu, wd)]
                    def f_m():
                        a = mx.quantized_matmul(x, *qm[0], transpose=True, mode=mode)
                        b = mx.quantized_matmul(x, *qm[1], transpose=True, mode=mode)
                        h = mx.sigmoid(a) * a * b
                        return mx.quantized_matmul(h, *qm[2], transpose=True, mode=mode)
                    row[mode + '_s'] = timeit(f_m, reps=3)
                except Exception as e:
                    row[mode + '_s'] = None; row[mode + '_err'] = str(e)[:120]
            out['results'].append(row); print(json.dumps(row), flush=True)
    a = mx.zeros((1 << 28,), dtype=mx.float32)
    dt = timeit(lambda: a + 1, reps=10)
    out['gpu_copy_GBps'] = 2 * a.nbytes / dt / 1e9
    return out

if __name__ == '__main__':
    print(json.dumps(run(), indent=1))
