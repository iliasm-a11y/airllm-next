import hashlib,json,os,struct,subprocess,sys
from pathlib import Path
import pytest

def test_real_strided_cache_hash_nan_and_zero_write(tmp_path):
    if sys.platform!='darwin':pytest.skip('Darwin/CommonCrypto native fixture')
    root=Path(__file__).parent;work=root.resolve().parents[2]
    src=Path(os.environ.get('AIRLLM_PROFILE_RUNTIME_SOURCE',work/'llama-v3'));lib=Path(os.environ.get('AIRLLM_PROFILE_RUNTIME_LIBRARIES',work/'runtime-v3-build/bin'))
    if not (lib/'libggml-cpu.dylib').exists():pytest.skip('pinned runtime required')
    binary=tmp_path/'state-fixture'
    done=subprocess.run(['clang++','-std=c++17','-O2','-Wno-deprecated-declarations','-I'+str(src/'ggml/include'),str(root/'state_fixture.cpp'),'-L'+str(lib),'-lggml','-lggml-base','-lggml-cpu','-Wl,-rpath,'+str(lib),'-o',str(binary)],capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    done=subprocess.run([str(binary)],capture_output=True,text=True);assert done.returncode==0,done.stderr
    a,b,c=[json.loads(x) for x in done.stdout.splitlines()]
    assert a['status']=='hashed' and a['finite'] and a['bytes']==24 and a['logical_bytes']==16
    assert a['sha256']==hashlib.sha256(struct.pack('<4f',1.,-0.,3.,4.)).hexdigest()
    assert b['status']=='hashed' and not b['finite']
    assert c['status']=='hashed' and c['bytes']==0 and c['logical_bytes']==0 and c['finite']
    assert c['sha256']==hashlib.sha256(b'').hexdigest()
