# 呼吸道H100长程无标签适配

基于dev28de0ee，继承已验收3b55cd5的方法与数值门控；只扩展可移植路径、普通LoRA入口、50轮日程/阶段提取与两H100队列。所有旧训练损失、LoRA/DoRA/Adapter数学保持。

`MCF_LONG_ROOT` 指向独立task根；`audit/manifest.json`封存源码、资产、方法和50轮预算。GPU0运行LoRA+→LoRA128，GPU1运行DoRA→Adapter256；每模型单卡。每10/20/30/50轮先严格检查完整前缀，提取/校验/直传后继续。CPU读出复用原始head，BIG最多6×8CPU/32GiB，16终点/1664拟合。

入口：`python -m micoformer.domain_long_adaptation.runner pipeline --slot 0`，需显式绑定获批H100UUID。数据/源码未封存、GPU占用或数值门控失败即不启动；没有scancel或资源取消逻辑。完整合同、执行与结果：项目tmp/20261024_respiratory_peft_mechanism_pilot/h100_long/。
