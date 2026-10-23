# 无标签领域 LoRA 实验入口

本分支从 dev@28de0ee 派生，只实现实验②的肠道/呼吸道无标签 LoRA 适配。
科学合同在 `../tmp/20261022_domain_adaptive_pretraining/PLAN.md`；状态以该实验README为准。

## 实现

- `micoformer/domain_adaptation/lora.py`：普通LoRA、48个packed QKV/output/FFN投影、冻结范围。
- `prepare.py`：精确复用V3.4领域名单，绑定已训练弱UniFrac50轮起点及source/asset hash。
- `workflow.py`：两组LR、10轮、checkpoint/RNG恢复、CUDA门控、frozen hash核验。
- `audit.py`：每域5条短真实样本的CPU前向/更新/下一步恢复/下游加载检查。
- `model_io.py`：依据manifest SHA严格加载适配后的完整模型，返回frozen eval编码器。
- `scripts/2.train_domain_lora.py`：prepare、verify、audit-cpu、submit、job；默认不执行任何阶段。

输入与计数加权的 `domain_pretraining/{data,input}.py` 是从实验① commit285e4d4独立复制的
已验证实现；训练runner由同提交workflow派生，只改变起点、LoRA/头参数范围、预算、门控和交付。
不从正在训练的实验worktree动态import，不修改dev/main或原实验。

普通LoRA不需要额外依赖。冻结的是原encoder权重，适配器会改变最终embedding；预测头同步更新。
full resume同时保存原权重、LoRA、自监督头、optimizer、scheduler、RNG、cursor和manifest SHA。
不将实验①的from-scratch状态或原始全语料optimizer当成本实验resume。

## 操作

在本worktree使用MiCoFormerV2 Python。先验证并commit，随后：

```bash
python scripts/2.train_domain_lora.py prepare --output-root ../tmp/20261022_domain_adaptive_pretraining
python scripts/2.train_domain_lora.py audit-cpu --manifest /absolute/path/lora_run_manifest.json --manifest-sha256 EXACT_SHA
python scripts/2.train_domain_lora.py verify --manifest /absolute/path/lora_run_manifest.json --manifest-sha256 EXACT_SHA
```

实际获准提交时使用同一manifest的submit命令；按respiratory_lora→gut_lora分配两张独立单卡任务。
提交入口先检查CPU验收，job入口再做GPU/真实batch/恢复门控，然后重新初始化正式运行。
资源为单A100、4CPU、24GiB、24小时；无scancel逻辑。提交回执逐job写入，重试跳过已提交项。

epoch10是主终点；epoch0/1/3/5及best为诊断。不据下游测试成绩改rank、LR、数据或epoch。
GPU smoke与CPU审计明确分开，CPU PASS不证明BF16、显存或实际学习率适用性。

## 三卡占位内执行（2026-10-23）

用户批准每组同机三卡，先呼吸道再肠道。`scripts/2.train_domain_lora_ddp.py`准备新版manifest并编排两个独立起点的DDP训练；`scripts/run_domain_lora_placeholder.sh`只启动工作层，不申请或结束allocation。每rank B128、全局384，原局部关系池和更新数保持；按目标数校正DDP均值。三rank RNG/optimizer/cursor严格恢复，每卡原门控及三卡恢复/验证汇总通过才训练。DataLoader使用forkserver；仅同拓扑恢复。旧单卡manifest和准备结果保留为历史，不将旧源码seal用于新执行。

B128 LoRA真实门控在40GB GPU达到显存上限后，新执行版本启用encoder现有non-reentrant激活重计算，保留dropout RNG、模型配置与所有科学参数；CPU核对前向/梯度/RNG，GPU重新通过门控再正式训练。旧失败路径和seal保留，新版本使用ddp3_recompute目录与独立manifest。
