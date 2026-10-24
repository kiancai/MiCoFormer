# Respiratory LoRA strength comparisons

Based on dev@28de0ee; reuses frozen-head implementation 46aa24a.
Approved stage: tmp/20261022_domain_adaptive_pretraining/strengthened_lora/.

Two independent starts from full-corpus weak-UniFrac epoch50:
- respiratory_lr3e4: rank16 / alpha32 / peak LR3e-4.
- respiratory_rank64: rank64 / alpha128 / peak LR1e-4.

All original encoder base weights and complete MLM/Query heads stay frozen.
Exact existing GOLD respiratory roster, seed42, ten epochs, three-rank B128
DDP, rank-ordered gradient reduction, activation recomputation. Each arm binds
its own initialization and parameter inventory. Rank16 initialization exactly
matches the earlier frozen-head baseline; both arms bind the same frozen hash.

Prepare with scripts/2.train_domain_lora.py prepare, then audit-cpu. Bind UUIDs
using scripts/2.train_domain_lora_ddp.py prepare, label ddp3_exact. Launch via
scripts/run_domain_lora_placeholder.sh inside preserved job673251 step_extern.
Each arm passes real-B128 single-rank/three-rank strict replay gates before
reloading the original initialization for formal training. Endpoint acceptance
precedes the next arm. Failures exit only working processes. The looping batch
script and allocation are never cancelled or altered by this implementation.

Formal runs bind committed source and immutable manifest/PLAN hashes. Downstream
uses frozen native mean embeddings and the existing Linear/MLP protocol at
fixed epoch10; the earlier rank16 baseline is reused without retraining.
