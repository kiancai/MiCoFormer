# 单卡领域语料训练

基于dev@28de0ee，在独立实验分支加入V3.3领域选择、6/12层配置和单卡梯度累积。历史三卡workflow及其冻结校验保持原样。科学合同与输出位于项目的`tmp/20261022_domain_corpus_pretraining/`。

四组为`gut_12l`、`gut_6l`、`respiratory_12l`、`respiratory_6l`，均seed42、512维、50epoch。每个作业申请A100一张，B128×3累积；UniFrac局部池仍是128，每项梯度按真实target计数归一。6层只放开层数限制，其余原配置校验沿用。

1. 使用MiCoFormerV2运行`python -m unittest discover -s tests/domain_pretraining -v`，检查后提交源码；不需安装pytest。
2. `python scripts/2.train_domain_mpc.py prepare --output-root <experiment>`生成不可变manifest；准备要求干净且已提交的worktree。
3. 对manifest求SHA256，以`submit --manifest <path> --manifest-sha256 <hash>`提交四个单卡排队任务。每次成功提交立即登记receipt；重复调用跳过已提交arm。
4. 作业内核验GPU归属，真实CUDA smoke验证B128累积、尾批、完整验证及恢复后的下一次更新，然后重新加载未训练epoch0并正式训练。smoke模型不会混入训练。

断点位于各arm的`run/last.pt`，包含optimizer/scheduler/RNG、epoch和下个microbatch游标；重提同一脚本时在四个位置参数后追加`--resume`。重提资源须在用户授权范围内进行；脚本不自动重提、不取消其他作业。每100步、每轮和终止预告时写checkpoint，固定epoch50为主要交付，best仅作诊断。

领域定义排除无法确认肠段的消化系统记录；呼吸道保留已确认系统但缺少更细部位者。沿用V3.3正式按行划分；manifest记录同Run跨split情况，不据此宣称Study/个体独立泛化。负样本观察词表从本领域train支持集重新计算，固定先验不重建。
