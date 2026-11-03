import unittest
from native_audit import compare_prompts


class NativeComparisonTests(unittest.TestCase):
    def test_rejects_omitted_prompts_and_token_mismatch(self):
        left = {"agents": [{"agent_id": 1, "rendered_prompt": "same", "token_ids": [1, 2]},
                           {"agent_id": 600, "rendered_prompt": "same", "token_ids": [1, 2]}]}
        self.assertFalse(compare_prompts(left, {"agents": left["agents"][:1]})["passed"])
        right = {"agents": [dict(a, token_ids=[1, 3]) for a in left["agents"]]}
        self.assertFalse(compare_prompts(left, right)["passed"])

    def test_rejects_render_or_identity_mismatch_even_with_same_ids(self):
        left = {"agents": [{"agent_id": 1, "rendered_prompt": "abc", "token_ids": [1]}]}
        for change in ({"agent_id": 600}, {"rendered_prompt": "xyz"}):
            right = {"agents": [dict(left["agents"][0], **change)]}
            self.assertFalse(compare_prompts(left, right)["passed"])

    def test_full_identity_establishes_token_prefix_only(self):
        data = {"agents": [{"agent_id": 1, "rendered_prompt": "abc", "token_ids": [1, 2]},
                           {"agent_id": 600, "rendered_prompt": "abd", "token_ids": [1, 3]}]}
        result = compare_prompts(data, data)
        self.assertTrue(result["passed"])
        self.assertEqual(result["native_common_token_ids"], [1])


if __name__ == "__main__":
    unittest.main()
