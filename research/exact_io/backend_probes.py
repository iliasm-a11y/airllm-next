"""Probe an explicitly supplied existing interpreter, without installing modules."""
import argparse
import json
from pathlib import Path
import subprocess
from common import save


def run(python):
    probes = {
        'torch_cpu': "import json, torch; a=torch.arange(8,dtype=torch.float32).reshape(2,4); "
                     "print(json.dumps({'torch':torch.__version__,'result':(a@a.T).tolist()}))",
        'mlx_cpu': "import json, mlx.core as mx; mx.set_default_device(mx.cpu); "
                   "a=mx.arange(8).reshape(2,4); print(json.dumps({'result':(a@a.T).tolist()}))",
    }
    results = {}
    for name, code in probes.items():
        try:
            result = subprocess.run([str(python), '-c', code], capture_output=True, text=True, timeout=30)
            results[name] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired:
            results[name] = dict(timed_out=True, timeout_seconds=30)
    value = dict(evidence='live subprocess import and arithmetic probes',
                 python_executable=str(python.resolve()), results=results,
                 packages_installed_by_experiment=False, model_downloaded=False)
    save('backend_probes.json', value)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    run(parser.parse_args().python)
