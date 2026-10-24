# 呼吸道无标签LoRA：随机与随机化长度组批

基于dev28de0ee；domain_adaptation/domain_pretraining复用已提交的5eac4cf源码，
本轮新增domain_bucket_adaptation与单卡入口，不改变旧实验或main/dev源码。

同一弱UniFrac epoch50起点、V3.4严格GOLD呼吸道名单、原先验与四项loss；
r128/alpha256/dropout0.05，原encoder与全部自监督头冻结。
两臂各10epoch、B128/累积1、8670更新；一张A100顺序执行。

random每轮打乱原train位置；window4096在同一随机序列的4096条窗口内按top512保留长度组批，
再打乱批次顺序。每轮完整覆盖。分桶自然改变负例候选、关系池与每批归一化的实际贡献，
本实验检验其总影响，不宣称逐步等价。验证统一沿原保存行序、stream2/epoch0固定批次。

准备、数值/恢复门控与正式执行分别使用：

```bash
python scripts/2.train_domain_bucket.py prepare
python scripts/2.train_domain_bucket.py gates --manifest MANIFEST --manifest-sha256 SHA
python scripts/2.train_domain_bucket.py sequence --manifest MANIFEST --manifest-sha256 SHA
```

后两步必须在673251的独立单A100 step内执行，使用run_domain_bucket_placeholder.sh配置环境。
正式执行要求已提交源码及manifest/hash一致、CUDA门控/测速PASS；支持显式--resume。
SIGUSR1/TERM/INT只保存和退出当前工作进程；代码不包含取消Slurm作业的动作。

主比较endpoint固定epoch10；best仅诊断，不用已看过的下游test重新挑epoch。
正式结果、配置和门控写入tmp/20261024_respiratory_peft_mechanism_pilot/a100_bucket。
