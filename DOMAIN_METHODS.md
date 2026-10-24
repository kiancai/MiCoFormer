# 呼吸道无标签适配方法筛选

dev28de0ee基线，独立实验worktree；继承45eeb40的已提交领域/分桶模块，旧运行树不改。
用户2026-10-25批准Adapter b256两LR、LoRA+ r64两A/B比例、DoRA r64两LR，单seed42。
完整合同与运行入口：`../tmp/20261024_respiratory_peft_mechanism_pilot/PLAN.md`及`design/METHOD_SCREEN_OVERNIGHT.md`。

- `domain_method_adaptation/modules.py`：新增参数化及严格可训练范围。
- `prepare.py`：每方法独立输入、初始化、版本与配置封存。
- `workflow.py`/`gates.py`：原分桶/分支执行、固定验证、完整恢复；GPU门控后正式各10轮。
- `controller.py`：逐卡等待前序两臂完成并退出，不修改/取消占位；失败按分支记录。
- `evaluation.py`：12个终点严格epoch10加载、同设备FP32探针、BF16提取/FP32保存。
- `scripts/domain_methods_cpu/`：BIG上既有呼吸道52fold×2head读出、完整重放、已验收geometry复用。
- `reporting.py`：回传hash与本地全部test指标复算、逐任务累积报告。

新增方法仅改变适配结构/优化器参数组；原encoder/stem/norm/prior和自监督头冻结。
统一V3.4严格GOLD呼吸道、原四loss、B128/window4096/10epoch；epoch10固定为主endpoint。
LoRA+各组调度保持比例；DoRA用逐输出行FP32/detach范数，bias不缩放、采用PEFT v0.10.0 dropout修正。
Adapter接在已有attention/FFN dropout之后、残差相加之前，GELU后新增dropout0.05，再升维。

CPU检查：`PYTHONPATH=. python -m unittest discover -s tests/domain_methods -v`。
准备/正式启动须绑定commit、manifest及GPU门控；最终训练、提取与CPU结果各自验收，不能以已排程代替完成。
