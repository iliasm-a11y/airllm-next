"""Bounded selection of unchanged GGUF Q8_0 matrices for matched GPU matvecs."""
import json
from pathlib import Path
import re


def plan(tail_layers=5, *, budget_bytes=3*1024**3):
    if type(tail_layers) is not int or not 0 <= tail_layers <= 45:
        raise ValueError('tail_layers must be 0..45')
    source = Path(__file__).parents[1] / 'glm53_m5/sources/gguf-headers.json'
    inventory = json.loads(source.read_text())
    selected = []
    for row in inventory['tensors']:
        name, dims = row['name'], row['dims']
        layer = re.match(r'blk\.(\d+)\.', name)
        if name != 'output.weight' and (not layer or int(layer[1]) < 45-tail_layers):
            continue
        if row['type'] != 'Q8_0' or len(dims) != 2 or dims[0] % 64 or dims[1] % 8:
            continue
        selected.append({'name': name, 'shape': dims, 'packed_bytes': dims[0]*dims[1]//32*34})
    weight_bytes = sum(row['packed_bytes'] for row in selected)
    if weight_bytes > budget_bytes:
        raise MemoryError('Selected GPU weights exceed fixed research admission budget')
    if not selected or 'output.weight' not in [row['name'] for row in selected]:
        raise ValueError('Pinned GGUF inventory has no expected output head')
    pattern = '^('+'|'.join(re.escape(row['name']) for row in selected)+')$'
    return {'schema_version': 1, 'tail_layers': tail_layers, 'selected_matrices': selected,
            'packed_weight_bytes': weight_bytes, 'max_weight_budget_bytes': budget_bytes,
            'tensor_pattern': pattern, 'model_revision': inventory['revision'],
            'scope': 'Static admission from pinned descriptors, not full working-set/RSS bound; only eligible single vectors use GPU'}
