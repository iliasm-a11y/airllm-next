"""Create a distinct fixture to measure steady graph reuse; freeze v1 intact."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parent/'metal-cpu-q8-bridge'
OLD=ROOT/'fixtures'; NEW=ROOT/'fixtures-warm-v2'
NEW.mkdir(exist_ok=False)
def change(value,old,new):
    assert value.count(old)==1,(old,value.count(old))
    return value.replace(old,new)
cpp=(OLD/'provider_fixture.cpp').read_text()
cpp=change(cpp,'const auto begin=std::chrono::steady_clock::now();\n        const auto status=ggml_backend_graph_compute(backend,graph);',
'''json iteration_seconds=json::array(),iteration_sha256=json::array();
        const auto begin=std::chrono::steady_clock::now();
        for(int iteration=0;iteration<5;++iteration) {
            const auto iteration_start=std::chrono::steady_clock::now();
            const auto status=ggml_backend_graph_compute(backend,graph);''')
cpp=change(cpp,'report["compute_wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();',
'''iteration_seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-iteration_start).count());''')
cpp=change(cpp,'{std::ofstream out(root/"output.f32",std::ios::binary);',
'''iteration_sha256.push_back(digest(y->data,y_bytes));
            if(iteration && iteration_sha256.back()!=iteration_sha256.front())throw std::runtime_error("graph reuse changed output bits");
        }
        report["compute_wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        report["iteration_seconds"]=iteration_seconds;report["iteration_sha256"]=iteration_sha256;
        report["repetitions"]=5;report["weight_residency_scope"]="same synthetic resident weights; no SSD or GLM speed claim";
        {std::ofstream out(root/"output.f32",std::ios::binary);''')
for label in ('cpu-mapped-q8-graph-v1',):cpp=cpp.replace(label,'cpu-mapped-q8-graph-warm-v2')
(NEW/'provider_fixture.cpp').write_text(cpp)
py=(OLD/'provider_fixture.py').read_text()
py=change(py,"return result\n\ndef logical_bound", "# v1 exceeded its unchanged128MiB cap on M8192/K4096/N64.\n    # These two larger resident shapes stay explicitly unqualified.\n    return [c for c in result if not (c['M']==8192 and c['K']==4096 and c['N'] in (64,128))]\n\ndef logical_bound")
py=py.replace('len(suite)==23','len(suite)==21')
py=change(py,"'processes_max':2*len(suite)","'repetitions_per_completed_graph':5,'first_iteration_includes_provider_initialization':True,\n        'warm_scope':'iterations1..4 reuse resident synthetic weights; not end-to-end inference',\n        'excluded_resource_shapes':['m8192-k4096-n64','m8192-k4096-n128'],\n        'processes_max':2*len(suite)")
py=change(py,"def execute(args):",'''def validate_repetitions(result):
    timings=result.get('iteration_seconds'); hashes=result.get('iteration_sha256')
    gc.require(result.get('repetitions')==5 and type(timings) is list and len(timings)==5
        and all(type(t) in (int,float) and math.isfinite(t) and t>=0 for t in timings),
        'FIXTURE_REUSE_TIMING','Five finite actual graph timings required')
    gc.require(type(hashes) is list and len(hashes)==5 and all(h==result.get('output_sha256') for h in hashes),
        'FIXTURE_REUSE_BITS','Every actual graph output must match the final raw output hash')
    return {'first_seconds':timings[0],'warm_seconds':timings[1:],
        'warm_median_seconds':sorted(timings[1:])[1:3][0]/2+sorted(timings[1:])[1:3][1]/2,
        'scope':'four resident graph repetitions, per-iteration checksum; no GLM/SSD performance claim'}

def execute(args):''')
# Both CPU and completed GPU branches must validate every repeated native result.
py=py.replace("job['result']=gc.read_json(root/'result.json')", "job['result']=gc.read_json(root/'result.json');job['reuse']=validate_repetitions(job['result'])")
py=change(py,"else:job['gpu_trace']=zero_trace(text,document,baseline['swap_used_bytes'],case['expectation']=='cpu_decline')",
"else:job['gpu_trace']=zero_trace(text,document,baseline['swap_used_bytes'],case['expectation']=='cpu_decline')\n                            if case['expectation']=='gpu':gc.require(job['gpu_trace']['published_operations']==5,'FIXTURE_REUSE_TRACE','Five actual final GPU publications required')")
(NEW/'provider_fixture.py').write_text(py)
build=(OLD/'build_fixture.py').read_text().replace("output=package/'provider-fixture'","output=package/'provider-fixture-warm-v2'")
(NEW/'build_fixture.py').write_text(build)
(NEW/'README.md').write_text('''# Distinct resident-graph measurement v2

The original 23-case suite remains failed with FIXTURE_RSS_LIMIT at138133504B.
This separate21-case suite excludes M8192/K4096/N64 and128; they are not qualified.
Each completed graph executes five times in the same process with the same
resident synthetic weights. The first operation includes provider initialization.
The following four operations are timed separately, with every output SHA256
required equal. Full CPU/GPU raw output and all native terminal/publication
traces remain checked. Expected deliberate fatal still stops on its first tile.
All RSS128MiB, logical64MiB, Metal64MiB, swap32MiB, original native/power/UID
guards and deadlines remain unchanged. No inference or SSD speed is measured.
''')
index={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in NEW.iterdir() if p.is_file()}
(NEW/'source-manifest.json').write_text(json.dumps({'scope':'distinct five-iteration resident graph candidate; v1 unchanged',
    'files_sha256':index,'original_source_sha256':{p:hashlib.sha256((OLD/p).read_bytes()).hexdigest() for p in ('provider_fixture.cpp','provider_fixture.py','build_fixture.py')},
    'model_loaded':False,'compiled':False,'gpu_executed':False},indent=2)+'\n')
print(json.dumps({'directory':str(NEW),'files':len(index),'original_sources_unchanged':True}))
