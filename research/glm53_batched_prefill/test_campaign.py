"""Portable lifecycle/admission fixtures; no GLM inference or performance claims."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from research.glm53_batched_prefill import campaign as c


def base_config():
    return {"threads": 10, "gpu_layers": 0, "batch_size": 8, "context_tokens": 1024,
            "model_path": "/fixture/model", "runtime_binary": "/fixture/server",
            "runtime_manifest_file": "/fixture/manifest", "verification_file": "/fixture/receipt"}


class Gate:
    passed = True
    def __init__(self, mode):
        assert mode == "ac"
    def sample(self, **kwargs):
        return {"passed": self.passed}
    def summary(self):
        return {"passed": self.passed}


class BatteryGate(Gate):
    passed = False


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps(base_config()))
    def tearDown(self):
        self.temp.cleanup()

    def selected(self):
        return c.qa.configuration(self.path, 256)[0]

    def test_adaptive_reduces_before_execution_and_keeps_unchanged_guards(self):
        config = self.selected()
        fixed = config["limits"]["reserve_bytes"] + config["limits"]["base_workspace_bytes"] + 1024 * 64 * 1024
        sample = {"available_bytes": fixed + 32 * 4 * c.MiB, "pressure_level": 1}
        result = c.adaptive_admission(config, 128, sample=sample)
        self.assertEqual(result["selected_microbatch"], 32)
        self.assertEqual(config["limits"]["max_rss_bytes"], int(8.5 * c.GiB))
        self.assertEqual(config["limits"]["max_swap_growth_bytes"], 32 * c.MiB)
        self.assertFalse(result["candidates"][0]["passed"])
        for pressure in (None, 2, 4):
            self.assertFalse(c.adaptive_admission(config, 8, sample={**sample,"pressure_level": pressure})["admitted"])

    def test_gpu_residence_counted_and_unsafe_profile_refused(self):
        config = self.selected()
        sample = {"available_bytes": 2 * c.GiB, "pressure_level": 1}
        self.assertTrue(c.adaptive_admission(config, 8, sample=sample)["admitted"])
        self.assertFalse(c.adaptive_admission(config, 8, gpu_layers=12, sample=sample)["admitted"])
        for invalid in (True, 4, 256):
            with self.assertRaises(ValueError):
                c.profile_config(config, invalid)
        with self.assertRaises(ValueError):
            c.profile_config(config, 32, 16)

    def test_logical_and_physical_command_sizes_are_distinct(self):
        model = object.__new__(c.BatchedResearchModel)
        model.research_microbatch = 16
        model.research_constructed_command = {}
        argv = ["fixture", "--batch-size", "128", "--ubatch-size", "128"]
        with mock.patch.object(c.NoCacheResearchModel, "_build_command", return_value=argv):
            selected = model._build_command(123)
        self.assertEqual(selected, ["fixture", "--batch-size", "128", "--ubatch-size", "16"])
        self.assertEqual(model.research_constructed_command["argv"], selected)

    def test_plan_records_unexecuted_and_no_invented_measurements(self):
        with mock.patch.object(c, "resource_snapshot", return_value={"available_bytes": 4*c.GiB,"pressure_level":1}):
            report = c.plan(self.path, gate_factory=Gate)
        self.assertEqual(report["status"], "planned")
        self.assertFalse(report["executed"])
        self.assertIsNone(report["measurements"])
        self.assertEqual(report["model_loads"], 0)
        self.assertEqual(report["planned_sequence"], [1,600,1])

    def test_battery_stops_before_runtime_or_native_help(self):
        factory = mock.Mock(side_effect=AssertionError("runtime must not start"))
        with mock.patch.object(c, "BENCHMARK_LOCK", self.root / "lock"), \
             mock.patch.object(c, "metadata_preflight", return_value={}), \
             mock.patch.object(c, "verify_cache_flags") as help_call:
            report = c.run_campaign(self.path, self.root / "run", runtime_factory=factory, gate_factory=BatteryGate)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["error_code"], "POWER_CONTEXT_UNAVAILABLE")
        self.assertEqual(report["requests"], [])
        self.assertFalse(report["prefill_measurements_complete"])
        factory.assert_not_called()
        help_call.assert_not_called()

    def test_prefill_mode_uses_full_prompts_but_cannot_claim_agent_completion(self):
        with mock.patch.object(c, "BENCHMARK_LOCK", self.root / "lock"), \
             mock.patch.object(c, "metadata_preflight", return_value={}):
            report = c.run_campaign(self.path, self.root / "prefill", 1, mode="prefill", gate_factory=BatteryGate)
        self.assertEqual(report["requested_tokens_each"], 1)
        self.assertIsNone(report["reasoning_effort"])
        self.assertFalse(report["functional_two_agents_qualified"])
        self.assertEqual([v["prompt"] for v in report["planned_agents"]], [c.TASKS[0]["prompt"],c.TASKS[1]["prompt"],c.TASKS[0]["prompt"]])

    def test_measurements_keep_unavailable_components_null(self):
        row={"prompt_token_ids":[1,2,3],"tokens":[9],"events":[{"t": 3.5,"response":{"tokens":[9]}}],
             "final":{"stop_type":"limit","timings":{"prompt_n":3,"prompt_ms":3000,"predicted_ms":500}}}
        result=c.measurement(row)
        self.assertEqual(result["prefill_seconds"],3)
        self.assertEqual(result["prompt_tokens_per_second"],1)
        self.assertEqual(result["ttft_seconds"],3.5)
        self.assertIsNone(result["gpu_compute_seconds"])
        self.assertIsNone(result["weight_wait_seconds"])
        self.assertFalse(c.measurement({})["timings_available"])
        self.assertIsNone(c.measurement({})["prefill_seconds"])

    def test_45_min_requests_cannot_raise_campaign_ceiling(self):
        with self.assertRaises(ValueError):
            c.run_campaign(self.path,self.root/"never",request_seconds=2700,max_seconds=10800)

    def test_three_request_lifecycle_preserves_native_reasoning_and_mission(self):
        class Awake:
            def __init__(self, owner): self.record={}
            def __enter__(self): return self
            def check(self): pass
            def close(self): pass
        class Process:
            pid=71001
            code=None
            def poll(self): return self.code
            def send_signal(self, unused): self.code=0
            def wait(self, **unused): return self.code
        process=Process()
        requests=[]
        memories={}
        current={}
        def http(method,path,payload=None,timeout=30):
            return {"prompt":current["marker"]} if path=="/apply-template" else {"tokens":[1,2,3]}
        def events(payload,cancel):
            yield {"content":current["answer"],"tokens":[9],"stop":False}
            yield {"content":"","tokens":[10],"stop":True,"stop_type":"eos","truncated":False,
                   "tokens_predicted":2,"timings":{"prompt_n":3,"prompt_ms":500,"predicted_n":2,"predicted_ms":1000}}
        model=SimpleNamespace(_process=process,_port=1,_samples=[],_http_json=http,_events=events)
        def status():
            return {"native":{"counters":{"loads":1,"requests":len(requests)},"scheduler":{"physical_limit":1},
                "checkpoint":{"integrity_verified":True},"runtime_artifacts_verified":True,"failure":None}}
        engine=SimpleNamespace(_model=model,load=lambda path:None,status=status,close=lambda:setattr(process,"code",0),request_shutdown=lambda:None)
        count=[0]
        def create(payload):
            count[0]+=1
            return {"id":"mission-%d"%count[0]}
        def dispatch(method,params):
            if method=="agent_identities": return list(range(600))
            if method=="agent_memory": memories[params["id"]]=params["text"]
            if method=="generate":
                requests.append(params)
                current["marker"]=next(task["marker"] for task in c.TASKS if task["agent_id"]==params["agent_id"])
                current["answer"]="</think>"+current["marker"]+ (" :17×24=408;420-408=12. Le stock suffit." if params["agent_id"]==1 else " : A=131;B=142;économie=11. Je choisis A.")
                model._http_json("POST","/apply-template",{})
                model._http_json("POST","/tokenize",{})
                list(model._events({"n_predict":256,"cache_prompt":False,"id_slot":0},None))
                return {"decision_id":"decision"}
            if method=="agent_decision": return {"messages":[{"role":"system","content":memories[params["id"]]}]}
            if method=="shutdown": engine.close()
            return {}
        runtime=SimpleNamespace(engine=engine,cancels={},store=SimpleNamespace(db=SimpleNamespace(close=lambda:None)),create=create,dispatch=dispatch)
        receipt=self.root/"receipt.json"
        receipt.write_text("{}")
        config=base_config();config["verification_file"]=str(receipt)
        self.path.write_text(json.dumps(config))
        observer=SimpleNamespace(snapshot=lambda:{"available":False})
        with mock.patch.object(c,"BENCHMARK_LOCK",self.root/"lock"), \
             mock.patch.object(c,"metadata_preflight",return_value={}), \
             mock.patch.object(c,"verify_cache_flags",return_value={}), \
             mock.patch.object(c,"resource_snapshot",return_value={"available_bytes":4*c.GiB,"pressure_level":1}), \
             mock.patch.object(c,"power_evidence_matches",return_value=True), \
             mock.patch.object(c.time,"sleep"), \
             mock.patch.dict(os.environ,{"RONIN_GLM_CONFIG":"previous"}):
            report=c.run_campaign(self.path,self.root/"sequence",runtime_factory=lambda path:runtime,
                gate_factory=Gate,awake_factory=Awake,io_factory=lambda pid:observer)
            self.assertEqual(os.environ["RONIN_GLM_CONFIG"],"previous")
        self.assertEqual([r["agent_id"] for r in requests],[1,600,1])
        self.assertTrue(all("reasoning_effort" not in r for r in requests))
        self.assertEqual(requests[0]["id"],requests[2]["id"])
        self.assertEqual(len(list((self.root/"sequence").glob("request-*/agent-*-events.jsonl"))),3)
        self.assertFalse(report["prefill_measurements_complete"])
        self.assertFalse(report["functional_checks"]["process_io_each"])
        self.assertFalse(report["exact_state_qualified"])
        self.assertFalse(report["adopted_optimization"])


class HardenedLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config_path = self.root / "config.json"
        receipt = self.root / "receipt.json"
        receipt.write_text("{}")
        config = base_config()
        config["verification_file"] = str(receipt)
        self.config_path.write_text(json.dumps(config))

    def tearDown(self):
        self.temp.cleanup()

    def exercise(self, *, mode="prefill", reasoning=None, power_valid=True,
                 exit_code=0, process_io=True, source_change=None):
        class Awake:
            def __init__(self, owner): self.record = {}
            def __enter__(self): return self
            def check(self): pass
            def close(self): pass
        class Process:
            pid = 71002
            code = None
            def poll(self): return self.code
            def send_signal(self, unused): self.code = exit_code
            def wait(self, **unused): return self.code
        process = Process()
        requests, memories, current = [], {}, {}
        budget = 1 if mode == "prefill" else 256
        def http(method, path, payload=None, timeout=30):
            return {"prompt": current["marker"]} if path == "/apply-template" else {"tokens": [1, 2, 3]}
        def events(payload, cancel):
            if mode == "prefill":
                yield {"content": "", "tokens": [9], "stop": True, "stop_type": "limit", "truncated": False,
                       "tokens_predicted": 1, "timings": {"prompt_n": 3, "prompt_ms": 500, "predicted_n": 1, "predicted_ms": 1000}}
            else:
                yield {"content": current["answer"], "tokens": [9], "stop": False}
                yield {"content": "", "tokens": [10], "stop": True, "stop_type": "eos", "truncated": False,
                       "tokens_predicted": 2, "timings": {"prompt_n": 3, "prompt_ms": 500, "predicted_n": 2, "predicted_ms": 1000}}
        model = SimpleNamespace(_process=process, _port=1, _samples=[], _http_json=http, _events=events)
        def status():
            return {"native": {"counters": {"loads": 1, "requests": len(requests)},
                "scheduler": {"physical_limit": 1}, "checkpoint": {"integrity_verified": True},
                "runtime_artifacts_verified": True, "failure": None}}
        def close():
            if process.code is None: process.code = exit_code
        engine = SimpleNamespace(_model=model, load=lambda path: None, status=status, close=close, request_shutdown=lambda: None)
        count = [0]
        def create(payload):
            count[0] += 1
            return {"id": "mission-%d" % count[0]}
        def dispatch(method, params):
            if method == "agent_identities": return list(range(600))
            if method == "agent_memory": memories[params["id"]] = params["text"]
            if method == "generate":
                requests.append(params)
                current["marker"] = next(task["marker"] for task in c.TASKS if task["agent_id"] == params["agent_id"])
                current["answer"] = "</think>" + current["marker"] + (" :17×24=408;420-408=12. Le stock suffit." if params["agent_id"] == 1 else " : A=131;B=142;économie=11. Je choisis A.")
                model._http_json("POST", "/apply-template", {})
                model._http_json("POST", "/tokenize", {})
                list(model._events({"n_predict": budget, "cache_prompt": False, "id_slot": 0}, None))
                if source_change and len(requests) == 1: source_change()
                return {"decision_id": "decision"}
            if method == "agent_decision": return {"messages": [{"role": "system", "content": memories[params["id"]]}]}
            if method == "shutdown": engine.close()
            return {}
        runtime = SimpleNamespace(engine=engine, cancels={}, store=SimpleNamespace(db=SimpleNamespace(close=lambda: None)), create=create, dispatch=dispatch)
        observer = SimpleNamespace(snapshot=lambda: {"available": process_io})
        with mock.patch.object(c, "BENCHMARK_LOCK", self.root / "lock"), \
             mock.patch.object(c, "metadata_preflight", return_value={}), \
             mock.patch.object(c, "verify_cache_flags", return_value={}), \
             mock.patch.object(c, "resource_snapshot", return_value={"available_bytes": 4*c.GiB, "pressure_level": 1}), \
             mock.patch.object(c, "power_evidence_matches", return_value=power_valid), \
             mock.patch.object(c.qa, "compare_process", return_value={"available": process_io}), \
             mock.patch.object(c.time, "sleep"):
            return c.run_campaign(self.config_path, self.root / "run", budget, mode=mode,
                reasoning_effort=reasoning, runtime_factory=lambda path: runtime,
                gate_factory=Gate, awake_factory=Awake, io_factory=lambda pid: observer)

    def test_prefill_accepts_complete_limit_without_eos_or_useful_final(self):
        report = self.exercise()
        self.assertTrue(report["prefill_measurements_complete"])
        self.assertFalse(report["functional_two_agents_qualified"])
        self.assertFalse(report["functional_checks"]["natural_eos_each"])
        self.assertFalse(report["functional_checks"]["useful_final_each"])
        self.assertTrue(all(report["prefill_admission_checks"].values()))
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["runner_source_identity"]["passed"])
        self.assertEqual(report["runner_source_sha256_at_start"], report["runner_source_identity"]["sha256_postflight"])

    def test_prefill_rejects_missing_power_continuity(self):
        report = self.exercise(power_valid=False)
        self.assertFalse(report["prefill_measurements_complete"])
        self.assertFalse(report["prefill_admission_checks"]["stable_ac"])
        self.assertEqual(report["status"], "rejected")

    def test_prefill_rejects_stopped_process_with_abnormal_exit(self):
        report = self.exercise(exit_code=-9)
        self.assertTrue(report["owned_server_stopped"])
        self.assertEqual(report["owned_server_returncode"], -9)
        self.assertFalse(report["prefill_measurements_complete"])
        self.assertFalse(report["prefill_admission_checks"]["normal_native_exit"])
        self.assertEqual(report["status"], "rejected")

    def test_prefill_rejects_missing_owned_process_io_evidence(self):
        report = self.exercise(process_io=False)
        self.assertFalse(report["prefill_measurements_complete"])
        self.assertEqual(report["status"], "rejected")

    def test_agent_native_reasoning_can_qualify_on_complete_fixture(self):
        report = self.exercise(mode="agents", reasoning=None)
        self.assertTrue(report["functional_two_agents_qualified"])
        self.assertTrue(report["functional_checks"]["native_reasoning_preserved"])

    def test_agent_explicit_max_reasoning_can_qualify_on_complete_fixture(self):
        report = self.exercise(mode="agents", reasoning="max")
        self.assertTrue(report["functional_two_agents_qualified"])

    def test_low_reasoning_cannot_qualify_agents(self):
        report = self.exercise(mode="agents", reasoning="low")
        self.assertFalse(report["functional_two_agents_qualified"])
        self.assertFalse(report["functional_checks"]["native_reasoning_preserved"])
        self.assertEqual(report["status"], "rejected")

    def test_high_reasoning_cannot_qualify_agents(self):
        report = self.exercise(mode="agents", reasoning="high")
        self.assertFalse(report["functional_two_agents_qualified"])
        self.assertFalse(report["functional_checks"]["native_reasoning_preserved"])
        self.assertEqual(report["status"], "rejected")

    def test_runner_source_change_invalidates_timing(self):
        source = self.root / "runner-source.py"
        source.write_text("# frozen start\n")
        with mock.patch.object(c, "__file__", str(source)):
            report = self.exercise(source_change=lambda: source.write_text("# changed during run\n"))
        self.assertFalse(report["runner_source_identity"]["passed"])
        self.assertFalse(report["prefill_measurements_complete"])
        self.assertEqual(report["error_code"], "RUNNER_SOURCE_CHANGED")
        self.assertEqual(report["status"], "failed")

    def test_runner_source_disappearance_invalidates_timing(self):
        source = self.root / "runner-source.py"
        source.write_text("# frozen start\n")
        with mock.patch.object(c, "__file__", str(source)):
            report = self.exercise(source_change=source.unlink)
        self.assertFalse(report["runner_source_identity"]["passed"])
        self.assertFalse(report["prefill_measurements_complete"])
        self.assertEqual(report["error_code"], "RUNNER_SOURCE_CHANGED")
        self.assertEqual(report["status"], "failed")


if __name__=="__main__":
    unittest.main()
