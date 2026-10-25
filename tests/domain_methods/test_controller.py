import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from micoformer.domain_method_adaptation.controller import predecessor_ready


class DependencyTest(unittest.TestCase):
    def fixture(self, root):
        (root/'audit').mkdir()
        (root/'audit/manifest.json').write_text('{}\n')
        digest=hashlib.sha256((root/'audit/manifest.json').read_bytes()).hexdigest()
        (root/'audit/manifest.sha256').write_text(digest+'\n')
        (root/'results').mkdir()
        for name in ['random','window4096']:
            run=root/'runs'/name
            (run/'results').mkdir(parents=True);(run/'run').mkdir()
            (run/'run/epoch10.pt').write_bytes(b'placeholder; strict tensor checks run in the GPU child')
            (run/'results/training_status.json').write_text(json.dumps(dict(status='COMPLETED',epochs=10,global_step=8670,manifest_sha256=digest)))
            epochs=[dict(epoch=i,exact_train_row_coverage=True,frozen_weights_unchanged=True,validation=dict(exact_row_coverage=True)) for i in range(1,11)]
            (run/'results/epochs.jsonl').write_text('\n'.join(json.dumps(e) for e in epochs)+'\n')
        return digest

    def test_requires_both_arms_and_sequence_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);digest=self.fixture(root)
            self.assertFalse(predecessor_ready(root))
            (root/'results/sequence_status.json').write_text(json.dumps(dict(status='COMPLETED',manifest_sha256=digest)))
            self.assertTrue(predecessor_ready(root))
            p=root/'runs/window4096/results/training_status.json'
            state=json.loads(p.read_text());state['global_step']=8669;p.write_text(json.dumps(state))
            with self.assertRaises(RuntimeError):predecessor_ready(root)

    def test_rejects_modified_seal_and_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);digest=self.fixture(root)
            (root/'results/sequence_status.json').write_text(json.dumps(dict(status='COMPLETED',manifest_sha256=digest)))
            p=root/'runs/random/results/epochs.jsonl'
            epochs=[json.loads(s) for s in p.read_text().splitlines()];epochs[3]['exact_train_row_coverage']=False
            p.write_text('\n'.join(json.dumps(e) for e in epochs))
            with self.assertRaises(RuntimeError):predecessor_ready(root)
            (root/'audit/manifest.json').write_text('{"changed":true}\n')
            with self.assertRaises(RuntimeError):predecessor_ready(root)


if __name__=='__main__':unittest.main()
