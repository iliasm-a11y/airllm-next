"""Bounded real GGUF sample validation versus upstream Python and compiled GGML C."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,platform,statistics,time,importlib.metadata,subprocess,tempfile,sys
import numpy as np
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/"air_llm"))
from airllm.nextgen.gguf import GGUFReader,GGMLType,dequantize
import gguf
from gguf.quants import dequantize as upstream
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint',type=Path)
parser.add_argument('--output',type=Path,default=ROOT/'evidence-native-gguf'/'codec-reference.json')
args=parser.parse_args()
TARGET=args.output
TARGET.parent.mkdir(exist_ok=True)
MODEL=args.checkpoint
_build=tempfile.TemporaryDirectory(prefix='airllm-gguf-reference-')
_libpath=Path(_build.name)/'codec-reference.so'
_flags=['cc','-O2','-ffp-contract=off','-dynamiclib' if sys.platform=='darwin' else '-shared']
subprocess.run(_flags+['-fPIC',str(BASE/'ggml_codec_reference.c'),'-o',str(_libpath)],check=True)
lib=ctypes.CDLL(str(_libpath))
for name in ('dequantize_row_q4_K','dequantize_row_q6_K'):
 f=getattr(lib,name);f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64];f.restype=None

def c_reference(packed,kind,shape):
 out=np.empty(shape,dtype=np.float32)
 fn=lib.dequantize_row_q4_K if kind==GGMLType.Q4_K else lib.dequantize_row_q6_K
 fn(packed.ctypes.data,out.ctypes.data,out.size)
 return out

def timed(fn,repeats=20):
 fn(); samples=[]
 for _ in range(repeats):
  start=time.perf_counter(); result=fn(); samples.append(time.perf_counter()-start)
 return {'median_seconds':statistics.median(samples),'minimum_seconds':min(samples),'repeats':repeats}

result={'schema_version':1,'model_path':str(MODEL),'model_file_bytes':MODEL.stat().st_size,
 'platform':platform.platform(),'numpy':np.__version__,'gguf_python':importlib.metadata.version('gguf'),
 'method':'pread only; no mmap; <=16 MiB encoded tensor payload across all samples; FP32 elementwise exact parity',
 'timing_scope':'Warm in-memory CPU codec microbenchmark; other team tests may run; not inference speed or physical SSD bandwidth',
 'provenance':{'ggml_source_url':'https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.c',
  'ggml_source_sha256':'5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab',
  'extracted_c_sha256':hashlib.sha256((BASE/'ggml_codec_reference.c').read_bytes()).hexdigest(),
  'python_upstream_source_url':'https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/quants.py',
  'python_upstream_download_sha256':'5c30d83fda855283a3239e479a26f35ea002ddd7650fcc4d255e6210fc9e72f1',
  'installed_gguf_quants_sha256':hashlib.sha256(Path(gguf.quants.__file__).read_bytes()).hexdigest(),
  'compiler_flags':_flags+['-fPIC'],
  'license':'MIT (GGML/llama.cpp); functions extracted unmodified, local half-to-float adapter uses _Float16'},
 'samples':[],'microbenchmarks':[]}
encoded_total=0
with GGUFReader(MODEL,include_tokenizer_metadata=False,max_read_bytes=1<<20) as reader:
 for kind in GGMLType:
  tensors=[t for t in reader.tensors.values() if t.quant_type==kind]
  chosen=[tensors[i] for i in sorted({0,len(tensors)//2,len(tensors)-1})]
  for tensor in chosen:
   for row in sorted({0,tensor.rows//2,tensor.rows-1}):
    packed=reader.read_packed_rows(tensor.name,row,1);encoded_total+=packed.nbytes
    actual=reader.read_rows(tensor.name,row,1);encoded_total+=packed.nbytes
    expected=upstream(packed,gguf.GGMLQuantizationType(int(kind))).reshape(actual.shape)
    py_exact=np.array_equal(actual,expected);c_exact=None;c_error=None
    if kind in (GGMLType.Q4_K,GGMLType.Q6_K):
     native=c_reference(packed,kind,actual.shape);c_exact=bool(np.array_equal(actual,native));c_error=float(np.max(np.abs(actual-native)))
    result['samples'].append({'name':tensor.name,'row':row,'tensor_shape':tensor.shape,'type':kind.name,
      'encoded_bytes':packed.nbytes,'elements':actual.size,'packed_sha256':hashlib.sha256(packed).hexdigest(),
      'python_exact':bool(py_exact),'python_max_abs_error':float(np.max(np.abs(actual-expected))),
      'c_exact':c_exact,'c_max_abs_error':c_error})
    assert py_exact and c_exact is not False
    del packed,actual,expected

  if kind in (GGMLType.Q4_K,GGMLType.Q6_K):
   tensor=tensors[0];count=min(256,tensor.rows,(1<<20)//tensor.row_bytes)
   packed=reader.read_packed_rows(tensor.name,0,count);encoded_total+=packed.nbytes
   assert encoded_total <= 16<<20
   raw=packed.tobytes();shape=(count,tensor.width)
   timings={
    'portable_numpy':timed(lambda:dequantize(raw,kind)),
    'upstream_python':timed(lambda:upstream(packed,gguf.GGMLQuantizationType(int(kind)))),
    'upstream_c':timed(lambda:c_reference(packed,kind,shape))}
   result['microbenchmarks'].append({'type':kind.name,'name':tensor.name,'rows':count,
    'encoded_bytes':packed.nbytes,'elements':count*tensor.width,'timings':timings})
   del packed,raw
 result['reader_stats']=dict(reader.stats)
 result['indexed_tensor_count']=len(reader.tensors)
 result['architecture']=reader.metadata.get('general.architecture')
result['encoded_sample_bytes_including_rereads']=encoded_total
assert encoded_total<=16<<20
result['status']='PASS'
TARGET.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'samples':len(result['samples']),'bytes':encoded_total,'target':str(TARGET)}))
