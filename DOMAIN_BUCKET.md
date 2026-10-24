# 呼吸道组批对照：两卡增补

本工作树从dev28de0ee建立，完整复用d501324已验证实现；原训练工作树保持不动。
新增r128/seed43随机与分桶两臂，用于复验采样方向；r64/seed42两臂用于与原r128/seed42匹配容量比较。
两臂内部除组批与其派生变化外相同，均10epoch/B128/累积1/8670更新，原V3.4/弱UniFrace50/固定头。
各pair初始化独立封存，全部验证题目固定seed42。训练seed控制LoRA初始化、corruption、组批与dropout。

新增任务每卡一个独立pipeline，在同step内依次数值/恢复门控、短测、随机训练与分桶训练。
每个新step请求3CPU，torch/BLAS单线程、DataLoader2 workers；原step的8CPU/4workers不变。
Slurm step可见三张已分配卡时，必须显式按其中的空闲UUID设置CUDA_VISIBLE_DEVICES；预检只查询目标卡，
并拒绝使用原卡3c1bc64f。该资源设置差异限制跨pair纯性能归因，不改变两臂内部配方比较。

CPU准备：显式设置MCF_PAIR=r128_seed43或r64_seed42，执行2.train_domain_bucket.py prepare。
GPU执行：run_domain_bucket_placeholder.sh WORKTREE pipeline PAIR TARGET_UUID --manifest PATH --manifest-sha256 SHA。
每pair有独立manifest、RUN目录、锁、进度与endpoint。停止工作不控制673251占位生命周期。

完整合同在tmp/20261024_respiratory_peft_mechanism_pilot/design/A100_PARALLEL_FOLLOWUPS.md。
