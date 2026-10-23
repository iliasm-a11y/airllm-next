#!/usr/bin/env python3
"""Export a committed research state, its restorable history and Python packages.

Only Git-tracked bytes enter the source archive. Local model downloads, caches,
environments and uncommitted artifacts are never scanned into the delivery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--ref', default='HEAD')
    parser.add_argument('--publication-receipt', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.is_relative_to(root):
        parser.error('Use an output directory outside the source checkout')
    output.mkdir(parents=True, exist_ok=True)
    def git(*values, **kwargs):
        return subprocess.check_output(['git', '-C', str(root), *values], **kwargs)
    commit = git('rev-parse', '--verify', args.ref+'^{commit}', text=True).strip()
    tree = git('rev-parse', commit+'^{tree}', text=True).strip()
    base = '9629c27b9d8bdf697ae866498ad468e11eb183d4'
    subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', base, commit], check=True)
    stem = 'AirLLM-Next-5.0.0a1'
    source = output/(stem+'-source.zip')
    subprocess.run(['git', '-C', str(root), 'archive', '--format=zip',
                    '--prefix=airllm-next/', '--output='+str(source), commit], check=True)
    bundle = output/(stem+'.bundle')
    subprocess.run(['git', '-C', str(root), 'bundle', 'create', str(bundle), '--all'], check=True)
    subprocess.run(['git', '-C', str(root), 'bundle', 'verify', str(bundle)],
                   check=True, capture_output=True)
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = {'version': '5.0.0a1', 'stage': 'research alpha',
               'source_commit': commit, 'source_tree': tree, 'base_commit': base,
               'branch': 'research/airllm-next-5-generation',
               'no_model_weights_included': True,
               'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)}
                         for p in (source, bundle)]}
    restore = (
        'AirLLM Next 5.0.0a1 — research alpha\n\n'
        'Extract the outer ZIP. For a normal source checkout, extract the source ZIP.\n'
        'For complete Git history:\n'
        f'  git clone {bundle.name} restored-airllm-next\n'
        '  git -C restored-airllm-next switch research/airllm-next-5-generation\n\n'
        f'Validated base: {base}\nDelivery source: {commit}\nGit tree: {tree}\n\n'
        'Read FINAL_VALIDATION_V5.md before use. Model weights are not included.\n'
        'Full GLM-5.3/Qwen3.8 generation on a16GB M5 is not certified.\n'
        'Install a Python package with its [next] extra, or follow the source README.\n')
    archive = output/(stem+'.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in (source, bundle):
            zipped.write(path, path.name)
        zipped.writestr('DELIVERY.json', json.dumps(receipt, indent=2)+'\n')
        zipped.writestr('RESTORE.txt', restore)
        if args.publication_receipt:
            zipped.write(args.publication_receipt, 'GITHUB_PUBLICATION.json')
        packages = git('ls-tree', '-r', '--name-only', commit, 'dist-v5', text=True).splitlines()
        for name in packages:
            if name.endswith(('.whl', '.tar.gz')):
                zipped.writestr(Path(name).name, git('show', commit+':'+name))
    for path in (source, archive):
        with zipfile.ZipFile(path) as zipped:
            if zipped.testzip() is not None:
                raise RuntimeError('Delivery ZIP integrity check failed')
    report = receipt | {'archive': str(archive), 'archive_bytes': archive.stat().st_size,
                         'archive_sha256': sha(archive)}
    (output/'delivery-receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
