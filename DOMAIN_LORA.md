# 冻结自监督头的无标签领域 LoRA 对照

基于 dev@28de0ee，完整复用已验收 LoRA 源码7711944，再仅改变训练范围：
MLM完整MLP、Query共享trunk及两个输出全部requires_grad=False；保留经头到encoder的autograd。
只有48个LoRA模块的A/B进入AdamW，head LR记录为0。原encoder、MLM/Query均纳入冻结hash。
不改数据、初始化seed、模型前向、rank16/alpha32、loss权重、LoRA LR、10轮或三卡B128。
新合同：../tmp/20261022_domain_adaptive_pretraining/frozen_heads/PLAN.md。

先prepare、audit-cpu，再用2.train_domain_lora_ddp.py prepare --execution-label ddp3_exact封印三卡manifest。
每域每卡CUDA/真实batch门控、三卡严格恢复、验证汇总通过，才重新加载原全语料起点训练。
sequence自动先呼吸道2890步、终点验收后再独立初始化肠道7680步；保留占位673251。
不把旧LoRA终点作为新起点。固定epoch10为主终点。源与旧产物独立保留。
