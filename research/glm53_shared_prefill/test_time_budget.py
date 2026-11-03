import json,tempfile,unittest
from pathlib import Path
from research.glm53_shared_prefill import qualify_agents as qa
from research.glm53_shared_prefill.test_qualify_agents import config

class TimeBudgetTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.path=self.root/'config.json';self.path.write_text(json.dumps(config()))
 def tearDown(self):self.tmp.cleanup()
 def test_explicit_45_changes_only_request_time(self):
  original,_=qa.configuration(self.path,256);longer,_=qa.configuration(self.path,256,request_seconds=2700)
  self.assertEqual(original['limits']['request_timeout_seconds'],1800)
  self.assertEqual(longer['limits']['request_timeout_seconds'],2700)
  longer['limits']['request_timeout_seconds']=1800
  self.assertEqual(original,longer)
 def test_stricter_explicit_timeout_preserved(self):
  raw=config();raw['limits']={'request_timeout_seconds':900};self.path.write_text(json.dumps(raw));chosen,_=qa.configuration(self.path,256,request_seconds=2700)
  self.assertEqual(chosen['limits']['request_timeout_seconds'],900)
 def test_rejects_unbounded_budget_or_global_extension(self):
  for value in (True,0,2701,3600,float('inf')):
   with self.subTest(value=value),self.assertRaises(ValueError):qa.configuration(self.path,256,request_seconds=value)
  with self.assertRaises(ValueError):qa.run_campaign(self.path,self.root/'never-created',request_seconds=2700,max_seconds=7201)
  self.assertFalse((self.root/'never-created').exists())

 def test_reversed_order_retains_both_full_tasks_and_default_checks(self):
  from unittest import mock
  from research.glm53_shared_prefill.test_qualify_agents import BatteryGate
  with mock.patch.object(qa,"BENCHMARK_LOCK",self.root/"lock"),mock.patch.object(qa,"metadata_preflight",return_value={}):
   report=qa.run_campaign(self.path,self.root/"blocked",agent_order=(600,1),request_seconds=2700,gate_factory=BatteryGate)
  self.assertEqual([t["agent_id"] for t in report["planned_agents"]],[600,1])
  self.assertEqual(report["planned_agents"],[qa.TASKS[1],qa.TASKS[0]])
  self.assertTrue(qa.campaign_checks({"requests":[{"agent_id":600},{"agent_id":1}]})["two_distinct_agents"])
  for order in ((1,1),(600,),(True,600)):
   with self.assertRaises(ValueError):qa.run_campaign(self.path,self.root/"never",agent_order=order)
