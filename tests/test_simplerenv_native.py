"""Native adapter result persistence tests; run in the simulator environment."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = Path(os.environ.get("SIMPLERENV_NATIVE_ROOT", ROOT / "third_party/SimplerEnv-OpenVLA"))

class NativeResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("native_eval_test", ADAPTER / "main_quant.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def run_case(self, fail):
        m = self.module
        with tempfile.TemporaryDirectory() as temp:
            with patch("sys.argv", ["main_quant.py", "--model", "mock-checkpoint"]):
                args = m._parse_args()
            args.logging_dir = str(Path(temp) / "rollouts")
            tasks = ROOT / "configs/evaluation/simplerenv_bridge_smoke.json"
            args.simplerenv_tasks_json = str(tasks)
            results = [[True], RuntimeError("simulator failed")] if fail else [[True], [False], [True], [False]]
            with patch.object(m, "OpenVLAInference"), patch.object(m, "maniskill2_evaluator", side_effect=results):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "simulator failed"):
                        m._run_fp_eval(args)
                else:
                    m._run_fp_eval(args)
            result = json.loads((Path(temp) / "results.json").read_text())
            self.assertEqual(result["status"], "failed" if fail else "completed")
            self.assertEqual(result["total_episodes"], 1 if fail else 4)
            self.assertEqual(result["total_successes"], 1 if fail else 2)
            self.assertEqual(len(result["tasks"]), 1 if fail else 4)
            self.assertFalse((Path(temp) / "results.tmp").exists())

    def test_success_and_task_failure_counts(self):
        self.run_case(False)

    def test_runtime_error_preserves_completed_tasks(self):
        self.run_case(True)

if __name__ == "__main__":
    unittest.main()
