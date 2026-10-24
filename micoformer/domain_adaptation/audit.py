"""Bounded CPU numerical verification; never substitutes for a CUDA gate."""
import copy
import gc
import math
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch

from micoformer.mpc_unifrac import workflow as common
from micoformer.mpc_unifrac.data import FullSupportRowDataset, DualCorruptionCollator, SharedEpoch, batch_to_device
from micoformer.mpc_unifrac.support import _state_close
from .prepare import validate_manifest, load_origin, SUBMISSION_ORDER
from .lora import frozen_sha256
from .model_io import load_adapted_checkpoint
from .workflow import load_model, make_optimizer, update, checkpoint, restore, stamp, state_equal


@torch.inference_mode()
def embedding(model, batch):
    model.eval()
    args = batch_to_device(batch, torch.device("cpu"))
    return model.unmasked_representations(args["genus_ids"], args["rclr"], args["padding_mask"])[1]


def cpu_acceptance(path, manifest_sha):
    manifest = validate_manifest(path, manifest_sha)
    output = Path(path).parents[1]
    corpus = manifest["assets"]["corpus"]["path"]
    with h5py.File(corpus, "r") as data:
        richness = np.diff(data["X/indptr"][:])
    reports = {}
    for name in SUBMISSION_ORDER:
        arm = manifest["arms"][name]
        rows = np.load(arm["data"]["paths"]["train"], allow_pickle=False)
        chosen = rows[(richness[rows] >= 5) & (richness[rows] <= 24)][:5]
        if len(chosen) != 5:
            raise RuntimeError("not enough real short samples for bounded CPU verification")
        dataset = FullSupportRowDataset(corpus, chosen)
        samples = [dataset[i] for i in range(5)]
        collate = DualCorruptionCollator(
            observed_genus_ids=np.load(arm["data"]["observed_genus_ids"], allow_pickle=False),
            seed=42, stream=91, shared_epoch=SharedEpoch(0))
        group = [collate(samples[:2]), collate(samples[2:])]
        model = load_model(arm, manifest, torch.device("cpu"))
        original = load_origin(manifest["assets"]["prior"]["path"], manifest["assets"]["origin"]["path"])
        before = embedding(original, group[0]).clone()
        if not torch.equal(before, embedding(model, group[0])):
            raise RuntimeError("zero-delta LoRA embedding changed")
        original.eval(); model.eval()
        with torch.inference_mode():
            left = original(**batch_to_device(group[0], torch.device("cpu")))
            right = model(**batch_to_device(group[0], torch.device("cpu")))
        if not state_equal(left, right):
            raise RuntimeError("zero-delta self-supervised losses changed")
        del original
        frozen = frozen_sha256(model)
        optimizer, scheduler = make_optimizer(model, arm)
        common._seed_all(42)
        statistics, first = update(model, optimizer, scheduler, group, arm, torch.device("cpu"))
        if not any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if n.endswith(".lora_B")):
            raise RuntimeError("adapter did not update")
        with tempfile.TemporaryDirectory(prefix="cpu_resume_", dir=output / "audit") as temporary:
            cp = Path(temporary) / "resume.pt"
            common._atomic_torch(cp, checkpoint(model, optimizer, scheduler, arm, manifest_sha,
                torch.device("cpu"), epoch=0, cursor=2, global_step=1, best_score=math.inf, statistics=statistics))
            _, expected_update = update(model, optimizer, scheduler, group[::-1], arm, torch.device("cpu"))
            expected_state = {k: v.clone() for k, v in model.state_dict().items()}
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            expected_scheduler = copy.deepcopy(scheduler.state_dict())
            del model, optimizer, scheduler
            gc.collect()
            model = load_model(arm, manifest, torch.device("cpu"))
            optimizer, scheduler = make_optimizer(model, arm)
            restored = restore(cp, model, optimizer, scheduler, arm, manifest_sha, torch.device("cpu"))
            assert (restored["global_step"], restored["next_batch"]) == (1, 2)
            _, actual_update = update(model, optimizer, scheduler, group[::-1], arm, torch.device("cpu"))
            if (not state_equal(expected_state, model.state_dict())
                    or not state_equal(expected_optimizer, optimizer.state_dict())
                    or expected_scheduler != scheduler.state_dict() or expected_update != actual_update):
                raise RuntimeError("resume differs from uninterrupted CPU next update")
            if frozen_sha256(model) != frozen:
                raise RuntimeError("frozen weights changed")
            after = embedding(model, group[0]).clone()
            if torch.equal(before, after):
                raise RuntimeError("encoder embedding did not change after adapter updates")
            common._atomic_torch(cp, checkpoint(model, optimizer, scheduler, arm, manifest_sha,
                torch.device("cpu"), epoch=0, cursor=4, global_step=2, best_score=math.inf, statistics=statistics))
            consumer = load_adapted_checkpoint(cp, manifest_sha256=manifest_sha)
            if not torch.equal(after, embedding(consumer, group[0])):
                raise RuntimeError("standalone adapted model restore changed embedding")
            if any(p.requires_grad for p in consumer.parameters()):
                raise RuntimeError("downstream loader failed to freeze the encoder")
            del consumer, restored, expected_state, expected_optimizer
        reports[name] = dict(status="PASS", real_train_rows=chosen.tolist(), microbatches=[2, 3],
            zero_delta_embedding_exact=True, zero_delta_losses_exact=True, adapter_updated=True,
            encoder_embedding_changed=True, frozen_weights_unchanged=True,
            strict_resume_next_update_exact=True, standalone_loader_embedding_exact=True,
            first_update=first, second_update=actual_update)
        del model, optimizer, scheduler, dataset, samples, group
        gc.collect()
    result = dict(status="PASS", device="cpu", scope="real short-row implementation audit; not CUDA or learning-rate acceptance",
                  checked_at=stamp(), manifest_sha256=manifest_sha, arms=reports)
    common._atomic_json(output / "audit/cpu_acceptance.json", result)
    return result
