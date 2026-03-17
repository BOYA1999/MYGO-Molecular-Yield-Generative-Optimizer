import os
import sys
import unittest
import tempfile
import subprocess


class SmokeTests(unittest.TestCase):
    def test_import_core_modules(self):
        # Basic import smoke tests (no heavy runtime execution).
        # If core deps are missing, skip with a clear message.
        try:
            import easydict  # noqa: F401
        except Exception:
            self.skipTest("Missing core dependency `easydict`. Install root requirements.txt first.")

        try:
            import utils.misc  # noqa: F401
            import utils.reconstruct  # noqa: F401
            import models.diffusion  # noqa: F401
            import models.maskfill  # noqa: F401
        except ModuleNotFoundError as e:
            self.skipTest(f"Missing dependency for core imports: {e}. Install root requirements.txt first.")

    def test_admet_cli_runs_rule_based(self):
        # This test requires RDKit to be installed.
        try:
            from rdkit import Chem  # noqa: F401
        except Exception:
            self.skipTest("RDKit not installed; skip ADMET CLI smoke test")

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(repo_root, "预测评估", "evaluate_molecules.py")
        outdir = tempfile.mkdtemp(prefix="admet_reports_")

        # Run rule-based mode to avoid requiring model files.
        cmd = [sys.executable, script, "-s", "CCO", "-o", outdir, "--no-ml", "--name", "ethanol"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"ADMET CLI failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

