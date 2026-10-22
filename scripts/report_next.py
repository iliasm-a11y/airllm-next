#!/usr/bin/env python3
"""Build factual report data from saved tests/benchmarks, never fabricate runs."""
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET
import re
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'air_llm'))
from airllm.nextgen.limits import estimate_glm
from airllm.nextgen.tuning import telemetry
import mlx.core as mx
import torch,transformers

summary={}
for name in ('next','ultra','upstream'):
    path=root/'evidence-next'/f'tests-{name}.xml'
    if path.exists():
        doc=ET.parse(path).getroot();suites=list(doc.iter('testsuite'))
        summary[name]={k:sum(int(x.get(k,'0')) for x in suites) for k in ('tests','failures','errors','skipped')}
        log=(root/'evidence-next/logs'/f'tests-{name}.log').read_text()
        passed=re.search(r'(\d+) passed',log)
        subtests=re.search(r'(\d+) subtests passed',log)
        summary[name]['passed_main']=int(passed.group(1)) if passed else 0
        summary[name]['passed_subtests']=int(subtests.group(1)) if subtests else 0
summary['environment']={'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,
    'mlx':mx.__version__,'torch':torch.__version__,'transformers':transformers.__version__,
    'metal_available':mx.metal.is_available(),'telemetry':telemetry()}
summary['not_executed']=['M5 Metal/TensorOps','Fanxiang USB throughput','full GLM-5.3 inference','semantic quality of GLM-5.3 quantizations','MTP/speculative decoding']
(root/'evidence-next/validation-summary.json').write_text(json.dumps(summary,indent=2))
c=json.loads((root/'evidence-next/sources/config-glm.json').read_text());bounds=estimate_glm(c)
(root/'evidence-next/hardware-bounds.json').write_text(json.dumps(bounds,indent=2))
print(json.dumps(summary,indent=2))
