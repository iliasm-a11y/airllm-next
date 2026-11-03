import unittest

from prefix_audit import common_prefix, token_summary, generated_messages


class PrefixAuditTests(unittest.TestCase):
    def test_boundary_merge_uses_full_prompt_ids_not_text_prefix(self):
        # Identical text up to the last character can end in different BPE IDs.
        agents = [
            {"agent_id": 1, "messages": [{"content": "éRONIN-001"}],
             "rendered_prompt": "éRONIN-001", "token_ids": [3, 101]},
            {"agent_id": 600, "messages": [{"content": "éRONIN-600"}],
             "rendered_prompt": "éRONIN-600", "token_ids": [3, 202]},
        ]
        result = token_summary(agents, microbatch=2)
        self.assertEqual(result["common_token_ids"], [3])
        self.assertEqual(result["common_token_count"], 1)
        self.assertEqual(result["system_text_lcp_utf8_bytes"], 8)
        self.assertIsNone(result["largest_nonzero_full_microbatch_boundary_inside_lcp"])

    def test_lcp_stops_before_divergence_and_handles_shorter_sequence(self):
        self.assertEqual(common_prefix([[1, 2, 3], [1, 2], [1, 2, 4]]), [1, 2])
        self.assertEqual(common_prefix([[1, 2], [1, 3], [1, 2]]), [1])
        self.assertEqual(common_prefix([[1], []]), [])
        self.assertEqual(common_prefix([]), [])

    def test_microbatch_candidate_never_exceeds_lcp(self):
        agents = [{"agent_id": i, "messages": [{"content": "shared"}],
                   "rendered_prompt": "shared", "token_ids": [1, 2, 3, 4, i]}
                  for i in (5, 6)]
        result = token_summary(agents, microbatch=2)
        self.assertEqual(result["largest_nonzero_full_microbatch_boundary_inside_lcp"], 4)
        with self.assertRaises(ValueError):
            token_summary(agents, microbatch=0)

    def test_actual_prompt_constructor_keeps_full_identity_and_private_memory(self):
        agents = generated_messages((1, 600), {
            1: {"prompt": "Explique le calcul 17 × 24.", "marker": "ALPHA-AUDIT"},
            600: {"prompt": "Compare les offres A et B.", "marker": "OMEGA-AUDIT"},
        })
        first, second = agents
        self.assertNotEqual(first["mission_id"], second["mission_id"])
        system1, system600 = [agent["messages"][0]["content"] for agent in agents]
        self.assertIn("RONIN-001 (agent 1)", system1)
        self.assertIn("RONIN-600 (agent 600)", system600)
        self.assertIn("ALPHA-AUDIT", system1)
        self.assertNotIn("OMEGA-AUDIT", system1)
        self.assertIn("OMEGA-AUDIT", system600)
        self.assertNotIn("ALPHA-AUDIT", system600)
        self.assertEqual(len(common_prefix([system1, system600]).encode()), 869)
        self.assertEqual(first["messages"][-1]["content"], "Explique le calcul 17 × 24.")


if __name__ == "__main__":
    unittest.main()
