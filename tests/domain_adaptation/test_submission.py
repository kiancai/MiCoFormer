"""Submission order follows the user's domain priority, including retries."""
import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("domain_cli", ROOT / "scripts/2.train_domain_lora.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class SubmissionOrderTests(unittest.TestCase):
    def test_respiratory_first_despite_manifest_order_or_selected_argument_order(self):
        cases = [(None, ["mcflora-resp-lora", "mcflora-gut-lora"]),
                 (["gut_lora", "respiratory_lora"], ["mcflora-resp-lora", "mcflora-gut-lora"])]
        for selected, expected in cases:
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                path = output / "audit/manifest.json"
                path.parent.mkdir()
                path.write_text("{}")
                (path.parent / "cpu_acceptance.json").write_text(json.dumps(dict(status="PASS", manifest_sha256="sealed-test-sha")))
                # JSON's alphabetical arm order places gut first. It must not
                # control scheduler submission priority.
                manifest = {"source": {"commit": "sealed-test-commit"}, "arms": {
                    name: {"output": str(output / "runs" / name)} for name in
                    ["gut_lora", "respiratory_lora"]}}
                calls = []
                def accept(command, **kwargs):
                    calls.append(command)
                    return SimpleNamespace(returncode=0, stdout=f"{700000+len(calls)}\n", stderr="")
                with patch("micoformer.domain_adaptation.prepare.validate_manifest", return_value=manifest), \
                     patch.object(cli.subprocess, "run", side_effect=accept):
                    cli.submit(path, "sealed-test-sha", selected)
                    cli.submit(path, "sealed-test-sha", selected)
                actual = [next(x.split("=", 1)[1] for x in cmd if x.startswith("--job-name=")) for cmd in calls]
                self.assertEqual(actual, expected)
                self.assertTrue(all(command[-1] == str(ROOT) for command in calls))


if __name__ == "__main__":
    unittest.main()
