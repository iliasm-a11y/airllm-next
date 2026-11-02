"""Exercise guard lifecycle with /bin/sleep only; never starts a model."""
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import tempfile
import unittest

from run_guarded_probe import run


@unittest.skipUnless(sys.platform == "darwin", "macOS libproc guard")
class GuardTests(unittest.TestCase):
    def arguments(self, directory, *, timeout=10, sleep="0.2"):
        return SimpleNamespace(output=Path(directory) / "guard.json", command=["--", "/bin/sleep", sleep],
                               max_rss_gib=9, max_swap_mib=64, timeout_seconds=timeout, poll_seconds=.02)

    def test_owned_process_completes_and_has_os_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            report = run(self.arguments(temp))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["returncode"], 0)
            self.assertTrue(report["samples"])
            self.assertIn("storage_read_bytes", report["samples"][0])
            self.assertIn("cpu_seconds", report["samples"][0])
            with self.assertRaises(ProcessLookupError):
                os.kill(report["owned_pid"], 0)

    def test_timeout_stops_only_owned_process_and_reaps_it(self):
        with tempfile.TemporaryDirectory() as temp:
            report = run(self.arguments(temp, timeout=.04, sleep="2"))
            self.assertEqual(report["status"], "stopped_or_failed")
            self.assertEqual(report["guard_violation"], "timeout")
            self.assertNotEqual(report["returncode"], 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(report["owned_pid"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
