# 肠道单H100 LoRA+50轮

从dev28de0ee继承已验收b7afbe6长程实现；独立肠道入口：
`MCF_LONG_ROOT=<task> python -m micoformer.domain_gut_adaptation.runner pipeline`。
只允许GPU3 UUIDf412ca27，单独合同与manifest，不改呼吸道原worktree。

数学/优化/数据路径沿用长程LoRA+；工作流仅扩展独立GPU3白名单与可关闭的milestone提取，默认呼吸道行为保持。
肠道保存固定阶段checkpoint，当前不执行embedding或BIG拟合。准备完先本地commit，再封存源码/输入。
