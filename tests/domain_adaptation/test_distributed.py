import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from micoformer.domain_adaptation.distributed import rank_batches, numeric_close, ForkserverEpoch, rank_ordered_mean
from micoformer.mpc_unifrac.model import Output
from micoformer.mpc_unifrac.support import distributed_training_loss


def gradient_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=3)
    model = DDP(torch.nn.Linear(1, 1, bias=False).double())
    model.register_comm_hook(None, rank_ordered_mean)
    with torch.no_grad():
        model.module.weight.fill_(.7)
    counts = np.array([[7, 2, 4, 128], [3, 11, 22, 127], [1, 5, 10, 2]])
    weights = dict(mlm=1., continuous=.53, presence=.24, unifrac=.00005)
    losses = [model(torch.tensor([[float((rank+1)*(j+1))]], dtype=torch.float64)).square().mean() for j in range(4)]
    result = Output(*losses, *counts[rank].tolist(), losses[0].detach(), losses[0].detach())
    loss = distributed_training_loss(result, weights, 3, torch.device('cpu'))
    loss.backward()
    expected = sum(weights[name] * sum(
        (2 * .7 * ((r+1)*(j+1))**2) * counts[r,j] / counts[:,j].sum()
        for r in range(3)) for j,name in enumerate(weights))
    actual = float(model.module.weight.grad)
    if not np.isclose(actual, expected, atol=2e-6, rtol=2e-7):
        raise AssertionError((rank, actual, expected))
    Path(output, f'rank{rank}').write_text(str(actual))
    dist.destroy_process_group()


def resume_worker(rank, rendezvous, output):
    import copy
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=3)
    def make():
        model = DDP(torch.nn.Sequential(torch.nn.Linear(5, 9), torch.nn.Dropout(.1),
                                       torch.nn.GELU(), torch.nn.Linear(9, 1)))
        model.register_comm_hook(None, rank_ordered_mean)
        return model, torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = make()
    def step():
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(4+rank, 5)
        model(x).square().mean().backward()
        optimizer.step()
    torch.manual_seed(42+rank)
    step()
    saved = copy.deepcopy((model.module.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
    step()
    expected = copy.deepcopy(model.module.state_dict())
    del model, optimizer
    model, optimizer = make()
    model.module.load_state_dict(saved[0])
    optimizer.load_state_dict(saved[1])
    torch.set_rng_state(saved[2])
    step()
    if not all(torch.equal(expected[k], v) for k,v in model.module.state_dict().items()):
        raise AssertionError('DDP rebuild changed next update')
    Path(output, f'rank{rank}').write_text('PASS')
    dist.destroy_process_group()


def epoch_worker(epoch, output):
    Path(output).write_text(str(epoch.value))


class DistributedTests(unittest.TestCase):
    def test_epoch_semaphore_matches_forkserver(self):
        with tempfile.TemporaryDirectory(prefix='lora-epoch-test-') as root:
            epoch = ForkserverEpoch(0)
            epoch.set(7)
            path = os.path.join(root, 'epoch')
            process = mp.get_context('forkserver').Process(target=epoch_worker, args=(epoch, path))
            process.start()
            process.join(30)
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail('forkserver epoch worker hung')
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(Path(path).read_text(), '7')

    def test_exact_batches_and_resume(self):
        for size, steps in [(110868, 289), (294906, 768), (1025, 3)]:
            plans = [rank_batches(size, seed=42, epoch=2, shuffle=True, rank=r) for r in range(3)]
            self.assertEqual([len(b) for _,b in plans], [steps]*3)
            rows = np.concatenate([np.concatenate(b) for _,b in plans])
            np.testing.assert_array_equal(np.sort(rows), np.arange(size))
            for rank, (_, batches) in enumerate(plans):
                _, resumed = rank_batches(size, seed=42, epoch=2, shuffle=True, rank=rank, start_batch=3)
                self.assertEqual(resumed, batches[1:])
            self.assertTrue(all(2 <= len(b) <= 128 for _,bs in plans for b in bs))

    def test_invalid_padding_is_rejected(self):
        for size, cursor in [(129, 0), (385, 0), (110868, 1)]:
            with self.assertRaises(RuntimeError):
                rank_batches(size, seed=42, epoch=0, shuffle=True, rank=0, start_batch=cursor)

    def test_nested_validation_comparison(self):
        self.assertTrue(numeric_close({'x': {'a': .5}, 'n': 3}, {'x': {'a': .5000001}, 'n': 3}))
        self.assertFalse(numeric_close({'x': .5}, {'x': .51}))

    def test_three_process_unequal_target_gradients(self):
        with tempfile.TemporaryDirectory(prefix='lora-ddp-test-') as root:
            mp.spawn(gradient_worker, args=(os.path.join(root, 'rendezvous'), root), nprocs=3, join=True)
            values = [float(Path(root, f'rank{i}').read_text()) for i in range(3)]
            self.assertEqual(values, [values[0]]*3)

    def test_three_process_exact_resume_after_bucket_rebuild(self):
        with tempfile.TemporaryDirectory(prefix='lora-ddp-resume-') as root:
            mp.spawn(resume_worker, args=(os.path.join(root, 'rendezvous'), root), nprocs=3, join=True)
            self.assertEqual([Path(root, f'rank{i}').read_text() for i in range(3)], ['PASS']*3)


if __name__ == '__main__':
    unittest.main()
