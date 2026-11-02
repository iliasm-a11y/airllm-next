"""Static patch checks and an optional actual patched CPU matmul_id fixture.

Set AIRLLM_EXPERT_DROP_FIXTURE to the compiled fixture linked against the patched
runtime. No model or GPU executes; a temporary synthetic file is read-only mmap.
"""
from pathlib import Path
import json
import os
import re
import subprocess
import unittest


ROOT = Path(__file__).parent


class ExpertDropTests(unittest.TestCase):
    def test_patch_is_default_off_and_changes_no_arithmetic(self):
        patch = (ROOT / "patches/llama-expert-drop.patch").read_text()
        added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
        removed = [line for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")]
        self.assertFalse(removed)
        self.assertIn('getenv("AIRLLM_EXPERT_DROP_AFTER_USE")', added)
        self.assertIn('ids->ne[1] == 1 && drop_env && strcmp(drop_env, "1") == 0', added)
        self.assertEqual(added.count("ggml_barrier(params->threadpool)"), 2)
        self.assertIn("if (ith == 0)", added)
        self.assertIn((ROOT / "expert_drop_helper.inc").read_text().strip(), added)
        self.assertNotIn("vec_dot(", added)
        self.assertNotIn("GGML_OP_", added)

    def test_helper_rejects_writable_anonymous_and_nonmapped_weights(self):
        code = (ROOT / "expert_drop_helper.inc").read_text()
        self.assertIn('"CPU_Mapped"', code)
        self.assertIn("GGML_TYPE_Q2_K", code)
        self.assertIn("GGML_BACKEND_BUFFER_USAGE_WEIGHTS", code)
        self.assertIn("basic.protection & VM_PROT_WRITE", code)
        self.assertIn("!extended.external_pager", code)
        self.assertIn("MADV_DONTNEED", code)
        self.assertIn("errors=%zu first_errno=%d", code)

    def test_full_page_ranges_stay_inside_original_experts(self):
        page = 16384
        for start in (0, 1, page - 1, page, page + 1):
            for stride in (1, page - 1, page, 2 * page, 2752512):
                end = start + stride
                begin_aligned = start + (page - start % page) % page
                end_aligned = end - end % page
                if begin_aligned < end_aligned:
                    self.assertGreaterEqual(begin_aligned, start)
                    self.assertLessEqual(end_aligned, end)
                    self.assertEqual(begin_aligned % page, 0)
                    self.assertEqual(end_aligned % page, 0)

    @unittest.skipUnless(os.environ.get("AIRLLM_EXPERT_DROP_FIXTURE"), "set AIRLLM_EXPERT_DROP_FIXTURE for patched CPU fixture")
    def test_actual_cpu_matmul_id_outputs_bitexact_and_advice_observed(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AIRLLM_", "GGML_", "LLAMA_ARG_", "DYLD_"))}
        done = subprocess.run([os.environ["AIRLLM_EXPERT_DROP_FIXTURE"]], text=True,
                              capture_output=True, env=env, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        result = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertTrue(result["decode_outputs_bit_identical"])
        self.assertTrue(result["mapped_original_bytes_unchanged"])
        rows = re.findall(r"AIRLLM_EXPERT_DROP tensor=\S+ selected=(\d+) advised=(\d+) bytes=(\d+) skipped=(\d+) errors=(\d+) first_errno=(\d+)", done.stderr)
        self.assertEqual(len(rows), 2)
        self.assertEqual([int(r[0]) for r in rows], [2, 1])
        self.assertTrue(all(int(r[1]) > 0 and int(r[2]) > 0 and int(r[4]) == 0 for r in rows), done.stderr)
        prefill = done.stderr.split("AIRLLM_FIXTURE_PREFILL_BEGIN", 1)[1].split("AIRLLM_FIXTURE_PREFILL_END", 1)[0]
        self.assertNotIn("AIRLLM_EXPERT_DROP", prefill)


if __name__ == "__main__":
    unittest.main()
