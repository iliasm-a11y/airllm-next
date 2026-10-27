"""Read-only admission evidence, no inference or pressure relief actions."""
import json, math, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"air_llm"))
from airllm.nextgen.experiment_guard import system_snapshot, ExperimentLimits, admission
out=Path(sys.argv[1]); seconds=float(sys.argv[2]) if len(sys.argv)>2 else 600
if not math.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite duration required')
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:f.write('{}\n')
start=time.monotonic(); samples=[]
while True:
    snap=system_snapshot()
    try: admission(ExperimentLimits(),snap); allowed=True; reason=None
    except MemoryError as error: allowed=False; reason=str(error)
    samples.append(dict(elapsed_seconds=time.monotonic()-start,admitted=allowed,reason=reason,**snap))
    out.write_text(json.dumps(dict(status="observing" if time.monotonic()-start<seconds else "complete",samples=samples),indent=2)+"\n")
    if time.monotonic()-start>=seconds: break
    time.sleep(5)
