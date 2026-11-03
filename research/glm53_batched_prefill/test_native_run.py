"""Owned-process/admission fixtures only; no GLM context or inference."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from research.glm53_batched_prefill import native_run as n


def config():
    return {"model_path":"/fixture/model","runtime_binary":"/fixture/server","runtime_manifest_file":"/fixture/manifest",
        "verification_file":"/fixture/receipt","context_tokens":2048,"batch_size":8,"threads":10,"gpu_layers":0,
        "mmap_demand":True,"expert_prefetch":True,"dense_prefetch":True,"mmap_nocache":True}


class Gate:
    passed=True
    def __init__(self,expected): assert expected=="ac"
    def sample(self,**unused): return {"passed":self.passed}
    def summary(self): return {"passed":self.passed}

class Battery(Gate):
    passed=False

class Child:
    pid=71001
    returncode=None
    def __init__(self): self.sent=[]
    def poll(self): return self.returncode
    def send_signal(self,value):
        self.sent.append(value)
        self.returncode=-value
    def wait(self,**unused): return self.returncode

class Awake:
    def __init__(self,child): self.record={"fixture":True}
    def __enter__(self): return self
    def check(self): pass
    def close(self): pass


class NativeRunTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.path=self.root/"config.json"
        self.path.write_text(json.dumps(config()))
    def tearDown(self): self.temp.cleanup()
    def selected(self): return n.qa.configuration(self.path,512,request_seconds=2700)[0]

    def test_environment_excludes_inherited_native_toggles_and_secrets(self):
        with mock.patch.dict(os.environ,{"AIRLLM_DENSE_LOCK":"1","GGML_TEST":"1","LLAMA_ARG_CTX_SIZE":"9999",
                "DYLD_INSERT_LIBRARIES":"bad","LLAMA_API_KEY":"secret","RONIN_PREFILL_OTHER":"1"}):
            env,knobs=n.clean_environment(config(),True)
        for key in ("AIRLLM_DENSE_LOCK","GGML_TEST","LLAMA_ARG_CTX_SIZE","DYLD_INSERT_LIBRARIES","LLAMA_API_KEY","RONIN_PREFILL_OTHER"):
            self.assertNotIn(key,env)
        self.assertEqual(knobs["AIRLLM_DENSE_PREFETCH_MAX_COLS"],"1")
        self.assertEqual(knobs["AIRLLM_MMAP_NOCACHE"],"1")
        self.assertEqual(knobs["RONIN_PREFILL_DIAGNOSTIC"],"1")

    def test_snapshot_and_diagnostic_are_charged_to_admission(self):
        c=self.selected()
        sample={"available_bytes":4*n.GiB,"pressure_level":1}
        full=n.admit(c,8,"full",False,sample)
        restored=n.admit(c,8,"restore",True,sample)
        self.assertEqual(restored["required_bytes_estimate"]-full["required_bytes_estimate"],2*(256*n.MiB+n.SCRATCH_POLICY))
        with self.assertRaises(n.qa.QualificationError):
            n.admit(c,8,"restore",True,{**sample,"available_bytes":full["required_bytes_estimate"]})
        for pressure in (None,2,4):
            with self.assertRaises(n.qa.QualificationError):
                n.admit(c,8,"full",False,{**sample,"pressure_level":pressure})

    def test_guard_reuses_exact_native_swap_and_pressure_policy(self):
        c=self.selected();model=n.guard_model(n.NativeLimits(**c["limits"]))
        baseline={"available_bytes":4*n.GiB,"swap_used_bytes":100*n.MiB,"pressure_level":1}
        self.assertIsNone(n.NativeGlmModel._guard_resource_sample(model,{**baseline,"rss_bytes":n.GiB},baseline,1))
        self.assertEqual(n.NativeGlmModel._guard_resource_sample(model,{**baseline,"rss_bytes":n.GiB,"swap_used_bytes":133*n.MiB},baseline,2),"SWAP_LIMIT")
        self.assertEqual(n.NativeGlmModel._guard_resource_sample(model,{**baseline,"pressure_level":4},baseline,3),"MEMORY_PRESSURE_CRITICAL")

    def test_signal_refuses_changed_pid_lifetime(self):
        child=Child()
        target=SimpleNamespace(create_time=lambda:456,exe=lambda:"/fixture/helper",is_running=lambda:True)
        with self.assertRaises(n.qa.QualificationError):
            n.stop_owned(child,"/fixture/helper",{"create_time":123},process_factory=lambda pid:target)
        self.assertEqual(child.sent,[])
        target.create_time=lambda:123
        cleanup=n.stop_owned(child,"/fixture/helper",{"create_time":123},process_factory=lambda pid:target)
        self.assertEqual(cleanup["signals"],["SIGTERM"])
        self.assertEqual(child.sent,[signal.SIGTERM])
        self.assertTrue(cleanup["owned_server_stopped"])

    def test_battery_blocks_before_artifact_preparation_or_native_launch(self):
        factory=mock.Mock(side_effect=AssertionError("no native process"))
        with mock.patch.object(n.qa,"BENCHMARK_LOCK",self.root/"lock"),mock.patch.object(n,"prepare") as prepare:
            report=n.run(self.path,"/fixture/helper","/fixture/helper.cpp","/fixture/prompts","/fixture/lib",self.root/"run",gate_factory=Battery,popen_factory=factory)
        self.assertEqual(report["status"],"blocked")
        self.assertEqual(report["error_code"],"POWER_CONTEXT_UNAVAILABLE")
        self.assertIsNone(report["returncode"])
        self.assertFalse(report["snapshot_qualified"])
        prepare.assert_not_called();factory.assert_not_called()
        self.assertEqual(report,json.loads((self.root/"run/guard.json").read_text()))

    def test_actual_owned_lifecycle_records_guard_samples_and_raw_returncode(self):
        child=Child()
        target=SimpleNamespace(create_time=lambda:123,exe=lambda:"/fixture/helper",is_running=lambda:True)
        c=self.selected()
        before={"available_bytes":4*n.GiB,"swap_used_bytes":100*n.MiB,"pressure_level":1}
        def sample(pid=None):
            if pid is not None:
                child.returncode=0
                return {**before,"rss_bytes":n.GiB}
            return before
        metadata={"snapshot_manifest":{},"tracked_identities":{}}
        with mock.patch.object(n.qa,"BENCHMARK_LOCK",self.root/"lock"), \
             mock.patch.object(n,"prepare",return_value=(metadata,{"agents":[]})), \
             mock.patch.object(n,"resource_snapshot",side_effect=sample), \
             mock.patch.object(n,"ProcessIOObserver",return_value=SimpleNamespace(snapshot=lambda:{"available":False,"scope":"fixture"})), \
             mock.patch.object(n,"power_evidence_matches",return_value=True), \
             mock.patch.object(n.time,"sleep"):
            report=n.run(self.path,"/fixture/helper","/fixture/helper.cpp","/fixture/prompts","/fixture/lib",self.root/"run",
                gate_factory=Gate,awake_factory=Awake,popen_factory=lambda *args,**kwargs:child,process_factory=lambda pid:target)
        self.assertEqual(report["status"],"completed")
        self.assertEqual(report["returncode"],0)
        self.assertEqual(report["samples"][0]["rss_bytes"],n.GiB)
        self.assertEqual(report["samples"][0]["swap_growth_bytes"],0)
        self.assertIsNone(report["guard_violation"])
        self.assertFalse(report["generation_performed"])
        self.assertFalse(report["snapshot_qualified"])
        self.assertEqual(child.sent,[])
        self.assertTrue((self.root/"run/guard-samples.jsonl").exists())

    def test_resource_violation_stops_owned_child_and_preserves_code(self):
        child=Child()
        target=SimpleNamespace(create_time=lambda:123,exe=lambda:"/fixture/helper",is_running=lambda:True)
        before={"available_bytes":4*n.GiB,"swap_used_bytes":100*n.MiB,"pressure_level":1}
        metadata={"snapshot_manifest":{},"tracked_identities":{}}
        with mock.patch.object(n.qa,"BENCHMARK_LOCK",self.root/"lock"), \
             mock.patch.object(n,"prepare",return_value=(metadata,{"agents":[]})), \
             mock.patch.object(n,"resource_snapshot",side_effect=lambda pid=None:{**before,**({"rss_bytes":9*n.GiB} if pid else {})}), \
             mock.patch.object(n,"ProcessIOObserver",return_value=SimpleNamespace(snapshot=lambda:{"available":False})), \
             mock.patch.object(n,"power_evidence_matches",return_value=True):
            report=n.run(self.path,"/fixture/helper","/fixture/helper.cpp","/fixture/prompts","/fixture/lib",self.root/"run",
                gate_factory=Gate,awake_factory=Awake,popen_factory=lambda *args,**kwargs:child,process_factory=lambda pid:target)
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["guard_violation"]["code"],"RSS_LIMIT")
        self.assertEqual(report["returncode"],-signal.SIGTERM)
        self.assertEqual(child.sent,[signal.SIGTERM])

if __name__=="__main__":
    unittest.main()
