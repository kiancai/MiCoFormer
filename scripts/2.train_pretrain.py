"""
scripts/2.train_pretrain.py — MiCoFormer 预训练入口脚本

使用示例：

  # epoch 模式
  python scripts/2.train_pretrain.py \
      --h5ad_path data/processed/microbiome_dataset.h5ad \
      --train_indices_path data/processed/splits/train.npy \
      --val_indices_path data/processed/splits/val.npy \
      --budget_mode epoch --max_epochs 20 --val_interval_epochs 3

  # step 模式
  python scripts/2.train_pretrain.py \
      --h5ad_path data/processed/microbiome_dataset.h5ad \
      --train_indices_path data/processed/splits/train.npy \
      --val_indices_path data/processed/splits/val.npy \
      --budget_mode step --max_steps 15000 --val_interval_steps 500
"""

import argparse

import numpy as np
from lightning.pytorch.utilities import rank_zero_info

from micoformer.utils.train_utils import int_or_float
from micoformer.workflows.pretrain import PretrainRunConfig, run_pretrain_once


TAG = "[train_pretrain]"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiCoFormer Stage 0 Pretraining")

    # 0.输入与切分参数
    p.add_argument("--h5ad_path", type=str, required=True)
    p.add_argument("--train_indices_path", type=str, required=True)
    p.add_argument("--val_indices_path", type=str, required=True)

    # 1.模型版本开关
    # V4 R2:距离驱动的 attention bias
    #   none  : baseline,无 bias
    #   taxo  : 离散 7-bucket 查 varp['taxo_dist']
    #   phylo : 3 层 MLP 查 varp['phylo_dist'](V5 默认)
    p.add_argument("--bias_type", type=str, default="phylo", choices=["none", "taxo", "phylo"])
    p.add_argument("--phylo_mlp_hidden", type=int, default=64,
                   help="phylo bias MLP 隐藏层维度(仅 --bias_type phylo 时生效;V5 默认 64,3 层 MLP)")
    p.add_argument("--phylo_bias_no_last_bias", action="store_true",
                   help="关掉 PhyloDistBias 末层 Linear 的 bias 项(推荐;实测 bias 是 dead-weight,关掉让 weight 真去学距离依赖)")

    # Tree loss(distance-preservation 辅助损失,见 utils/tree_loss.py)
    p.add_argument("--tree_loss_weight", type=float, default=0.0,
                   help="Tree loss 系数 λ:loss_total = L_MLM + λ × (L_pair + L_triplet);默认 0=不开。推荐 0.1")
    p.add_argument("--tree_n_pairs", type=int, default=256,
                   help="每 forward 抽多少 token pair 算 L_pair(默认 256)")
    p.add_argument("--tree_n_triplets", type=int, default=128,
                   help="每 forward 抽多少 triplet(anchor + phylo 近邻 + phylo 远端;默认 128)")
    p.add_argument("--tree_margin", type=float, default=0.5,
                   help="Triplet margin:||h_a - h_p|| + margin < ||h_a - h_n|| 才不罚(默认 0.5)")

    # X2 多任务范式(2026-05-28 夜,详 decisions / roadmap §4.1 d)
    # phase 1 推荐:--mlm_weight 1.0 --x2_phylo_weight 1.0 --bias_type none(撤 attention bias)
    # phase 2 蛋白完成后:加 --x2_protein_weight 1.0 --use_protein_pe(需要 varm['protein_pe'])
    p.add_argument("--mlm_weight", type=float, default=1.0,
                   help="X2 多任务:abundance huber 回归权重,默认 1.0(0=关掉 MLM)")
    p.add_argument("--x2_phylo_weight", type=float, default=0.0,
                   help="X2 多任务:预测 phylo coord MSE 权重,默认 0(=旧 MLM 行为);phase 1 推荐 1.0")
    p.add_argument("--x2_protein_weight", type=float, default=0.0,
                   help="X2 多任务:预测 protein coord MSE 权重,默认 0(蛋白 phase 2 开,需 use_protein_pe)")
    p.add_argument("--x2_head_hidden", type=int, default=128,
                   help="PriorCoordHead 中间层维度,默认 128")
    p.add_argument("--use_protein_pe", action="store_true", default=False,
                   help="X2 phase 2:启用蛋白 PE 输入通道(要求 varm['protein_pe'] 已写入语料)")
    p.add_argument("--protein_pe_hidden", type=int, default=128,
                   help="ProteinPE 投影 MLP 中间维度,默认 128")

    # Phylo Soft-Target CE(2026-05-29,替代 X2 32d MSE — X2 实测 mean collapse)
    # 推荐:--phylo_ce_weight 1.0 --phylo_ce_tau 6.5 --bias_type none
    p.add_argument("--phylo_ce_weight", type=float, default=0.0,
                   help="Phylo Soft-Target CE loss 权重 — vocab_head + soft target(softmax(-dist/tau));"
                        "默认 0=关;新方案 phase 1 推荐 1.0")
    p.add_argument("--phylo_ce_tau", type=float, default=6.5,
                   help="Phylo Soft-Target CE 温度 τ:soft target = softmax(-dist/τ);"
                        "推荐 6.5 ≈ log1p(patristic_max=656),让近亲 vs 远亲 prob ratio 在 e^2~e^4")
    p.add_argument("--phylo_w_weight", type=float, default=0.0,
                   help="Phylo Tree-Wasserstein simplified loss 权重 — W-1 expected phylo distance;"
                        "默认 0=关;phase 2 推荐 1.0(无 hyperparameter,不 ep0 saturate)")
    # Protein Tree-Wasserstein simplified(2026-05-30,phylo_w 精确镜像,蛋白距离矩阵)
    p.add_argument("--protein_w_weight", type=float, default=0.0,
                   help="Protein Tree-Wasserstein simplified loss 权重 — W-1 expected protein distance;"
                        "默认 0=关;镜像 phylo_w,需 --protein_dist_path")
    p.add_argument("--protein_pe_path", type=str, default=None,
                   help="protein_feat.npy 外部路径([V_real, 480]),给 protein PE embedding 输入用;"
                        "use_protein_pe=True 时需要")
    p.add_argument("--protein_dist_path", type=str, default=None,
                   help="protein_dist.npy 外部路径([V_real, V_real] float32,对角=0、对称),"
                        "给 protein_w loss 用;protein_w_weight>0 时需要")
    # 对比学习(2026-06-04,InfoNCE,保留 MLM 锚)
    p.add_argument("--contrastive_weight", type=float, default=0.0,
                   help="InfoNCE 对比 loss 权重 — 同样本两套 abund-mask 两视图拉近;默认 0=关,典型 0.1-0.5")
    p.add_argument("--contrastive_temp", type=float, default=0.1, help="NT-Xent 温度 τ(典型 0.1)")
    p.add_argument("--contrastive_proj_dim", type=int, default=128, help="projection head 输出维度")
    p.add_argument("--contrastive_mask_prob", type=float, default=0.15, help="第二视图 abund-mask 比例")
    # JEPA(2026-06-04,潜空间预测被遮 genus 含义向量;红线:坐标只当地址 query,target 是含义向量)
    p.add_argument("--jepa_weight", type=float, default=0.0,
                   help="JEPA latent-prediction loss 权重;默认 0=关,典型 1.0。需 use_phylo_pe/use_protein_pe;"
                        "三卡须配 --ddp_find_unused_parameters(JEPA 模式 genus_mask_token 闲置)")
    p.add_argument("--jepa_mask_ratio", type=float, default=0.5,
                   help="target token 比例(从 context 移除、由 predictor 预测;典型 0.5)")
    p.add_argument("--jepa_mlm_mask_prob", type=float, default=0.15,
                   help="MLM 锚的 abund-mask 比例(在 context 内选;防塌锚,典型 0.15)")
    p.add_argument("--jepa_ema_decay", type=float, default=0.996,
                   help="target encoder EMA 起步衰减(0.996,训练中 linear ramp→1;I-JEPA)")
    p.add_argument("--jepa_pred_dim", type=int, default=256,
                   help="窄 bottleneck predictor 宽度(典型 256 = d_model 的 0.5x;bottleneck 是防塌主力)")
    p.add_argument("--jepa_pred_depth", type=int, default=2, help="predictor transformer 层数(典型 2)")
    p.add_argument("--jepa_pred_heads", type=int, default=4, help="predictor 注意力头数(pred_dim 须整除)")
    p.add_argument("--jepa_vicreg_weight", type=float, default=0.0,
                   help="VICReg variance 防塌正则权重(后备;起步 0,塌了抬,见 PLAN 防塌段)")
    p.add_argument("--jepa_mask_mode", type=str, default="structured",
                   choices=[
                       "random", "structured",
                       "structured_hi", "structured_hi_phylo", "structured_hi_protein",
                   ],
                   help="JEPA target 遮挡方式:random=随机遮 ~ratio;structured=按 phylo/protein 坐标"
                        "样本内成簇遮;structured_hi*=高丰度 token 作 seed 后沿 phylo/protein 扩块,"
                        "只把先验当出题人,不作 target/loss")
    p.add_argument("--jepa_addr_mode", type=str, default="coords", choices=["coords", "genus"],
                   help="JEPA address query:coords=phylo/protein 坐标(历史,已证'错图');"
                        "genus=被遮菌 genus 身份(Cell-JEPA 式,吃数据驱动菌间共变,2026-06-09)")
    p.add_argument("--jepa_n_seeds", type=int, default=4,
                   help="structured 模式多少个种子簇(I-JEPA multi-block;每簇遮 ratio/n_seeds 最近邻;v2 默认 4)")
    # JEPA v2(2026-06-06,删 MLM + 双自监督 + 防塌升级)
    p.add_argument("--jepa_global_weight", type=float, default=0.5,
                   help="JEPA v2 全局对齐 loss 权重(student PMA vs teacher PMA;默认 0.5,0=关)")
    p.add_argument("--jepa_n_reg_tokens", type=int, default=4,
                   help="JEPA v2 register token 数(T-JEPA 防塌,encoder 前缀;默认 4,0=关)")
    p.add_argument("--jepa_ratio_start", type=float, default=0.3,
                   help="JEPA v2 structured mask ratio curriculum 起点(epoch 0;默认 0.3)")
    p.add_argument("--jepa_ratio_end", type=float, default=0.5,
                   help="JEPA v2 structured mask ratio curriculum 终点(末 epoch;默认 0.5)")
    # JEPA v3(2026-06-11,全盘抄 GeneJEPA set 级范式)
    p.add_argument("--jepa_setlevel", action="store_true", default=False,
                   help="开纯 set 级 JEPA(GeneJEPA 式):student context→PMA→z_s,teacher 只看 target 子集→PMA→z_t,"
                        "predictor 对齐;砍 token 级 predictor。配 --jepa_mask_mode random")
    p.add_argument("--jepa_loss_type", type=str, default="cosine", choices=["cosine", "mse"],
                   help="set 级对齐 loss:cosine(GeneJEPA 式,默认)| mse(I-JEPA 式)")
    p.add_argument("--jepa_ema_end", type=float, default=0.9995,
                   help="setlevel EMA cosine 调度终点(GeneJEPA 0.9995;起点=jepa_ema_decay)")
    p.add_argument("--jepa_ema_warmup_steps", type=int, default=0,
                   help="setlevel EMA warmup:前 N 步 teacher 冻结(GeneJEPA 2000;0=不 warmup)")
    p.add_argument("--jepa_student_vicreg_weight", type=float, default=0.0,
                   help="setlevel student z_s 的 VICReg 权重(GeneJEPA 在 student_ctx 也加防塌)")
    p.add_argument("--jepa_predict_residual", action="store_true", default=False,
                   help="token 级 JEPA target 减样本全局中心(predict residual):逼预测被遮菌相对整体偏差,铲'看整体'捷径")

    # 去批次(2026-06-08;study=Project_ID;默认全关=与现状等价)
    p.add_argument("--study_balanced", action="store_true", default=False,
                   help="train 用 study-balanced 批采样(每 batch 同一 study,CONCORD 对比用)")
    p.add_argument("--use_study_conditioning", action="store_true", default=False,
                   help="条件 MLM 头:study_embed[study_id] 只进重建头(scVI 式),encoder/PMA 输出 batch-free")
    p.add_argument("--study_min_size", type=int, default=64,
                   help="样本数 >= 此值的 study 各占一 id,小尾巴并 UNK(0)")

    # V5 新增:三段相加 + PMA + metadata 多任务
    p.add_argument("--abundance_encoding", type=str, default="mlp", choices=["mlp", "bin"],
                   help="V5:abundance 输入编码方式;mlp=连续 MLP(默认),bin=旧离散 embedding")
    p.add_argument("--abundance_value_transform", type=str, default="rclr_sigma",
                   choices=["rclr_sigma", "rclr", "rank", "presence", "raw"],
                   help="V5 §4.2:present-only abundance 数值写法(编码消融);rclr_sigma=现状默认,"
                        "rclr=去σ,rank=排名,presence=只打勾(MLM退化、慎用),raw=相对丰度原值")
    p.add_argument("--abundance_loss", type=str, default="huber", choices=["huber", "bin_ce"],
                   help="V5:abundance MLM loss;huber=连续回归(默认),bin_ce=旧 bin 分类")
    p.add_argument("--no_phylo_pe", action="store_true", default=False,
                   help="V5:禁用 PhyloPE(默认启用)")
    p.add_argument("--phylo_pe_hidden", type=int, default=128,
                   help="PhyloPE 投影 MLP 中间维度,默认 128")
    p.add_argument("--pooling_mode", type=str, default="pma", choices=["pma", "mean_pool"],
                   help="V5:sample-level pooling;pma(默认) | mean_pool")
    p.add_argument("--pma_nhead", type=int, default=4)
    p.add_argument("--pma_k", type=int, default=1)
    p.add_argument("--no_metadata_task", action="store_true", default=False,
                   help="V5:禁用 EnvCategory 多任务监督(默认启用)")
    p.add_argument("--metadata_loss_weight", type=float, default=0.3,
                   help="V5:λ_meta(metadata loss 权重),默认 0.3")
    p.add_argument("--huber_beta", type=float, default=1.0)

    # 2.1.模型主体参数
    p.add_argument("--d_model", type=int, default=512)                # token embedding 的维度，也是模型中间层的维度
    p.add_argument("--nhead", type=int, default=16)                    # 多头注意力中的头数
    p.add_argument("--num_layers", type=int, default=12)               # Transformer Encoder 层数
    p.add_argument("--ff_dim", type=int, default=None,
                   help="FeedForward 绝对维度，与 --ff_ratio 互斥。不指定时使用 ff_ratio。")
    p.add_argument("--ff_ratio", type=int, default=None,
                   help="FeedForward 比例（dim_ff = d_model × ff_ratio），与 --ff_dim 互斥。默认 4。")
    p.add_argument("--num_abundance_bins", type=int, default=40)      # 丰度分箱数量

    # 2.2.模型主体参数的协议参数
    p.add_argument("--abundance_mode", type=str, default="abs_log_bins", choices=["abs_log_bins", "rank_bins"])
    p.add_argument("--min_abundance", type=float, default=4e-6)       # 最小丰度阈值
    p.add_argument("--max_seq_len", type=int, default=512)            # 每个样本保留的最大物种数 (截断长度);V5 主线一直用 512(cover 97.2%样本,p95=436);2026-05-29 default 1024→512(详 memory `seq_len_512_default`)

    # 3.1.预训练中的训练主体参数
    p.add_argument("--batch_size", type=int, default=32,
                   help="per-GPU micro-batch（DDP 下有效 batch = batch_size × devices × accumulate_grad_batches）")
    p.add_argument("--mask_prob", type=float, default=0.15)           # 预训练 Mask 概率
    p.add_argument("--dropout", type=float, default=0.1)              # Dropout 概率
    p.add_argument("--lr", type=float, default=3e-4)                  # 学习率
    p.add_argument("--weight_decay", type=float, default=1e-2)        # 权重衰减 (L2 正则化)
    p.add_argument("--warmup_ratio", type=float, default=0.02)        # Warmup 占总 optimizer steps 的比例

    # 3.2.预训练中的协议参数
    # lr_scheduler_type 决定学习率下降方式：
    # - cosine：warmup 后按 cosine 平滑衰减
    # - plateau：warmup 后根据 val/loss 是否停滞来自动降 LR
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["cosine", "plateau"])
    p.add_argument("--lr_plateau_factor", type=float, default=0.5)       # plateau 降学习率的乘法因子
    p.add_argument("--lr_plateau_patience", type=int, default=2)         # plateau 在多少次验证无改善后降 LR
    p.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)      # plateau 的最小学习率

    # 4. 预算与验证协议参数
    # budget_mode 决定"训练预算"的单位：
    # - epoch：更适合当前这种数据规模不算特别大的实验
    # - step：更适合超大数据集或只想固定 optimizer 更新次数的场景
    p.add_argument("--budget_mode", type=str, default="epoch", choices=["epoch", "step"])
    p.add_argument("--max_epochs", type=int, default=None)             # epoch 模式下的最大训练轮数
    p.add_argument("--max_steps", type=int, default=None)              # step 模式下的最大训练步数
    p.add_argument("--val_interval_epochs", type=int, default=None)    # epoch 模式下每多少个 epoch 验证一次
    p.add_argument("--val_interval_steps", type=int, default=None)     # step 模式下每多少步验证一次
    p.add_argument("--limit_train_batches", type=int_or_float, default=1.0)   # 每 Epoch 仅使用部分训练数据 (float=比例 / int=绝对 batch 数)
    p.add_argument("--limit_val_batches", type=int_or_float, default=1.0)     # 每 Epoch 仅使用部分验证数据 (float=比例 / int=绝对 batch 数)

    # 4.1. Early stopping（0=禁用，与 finetune 对称）
    p.add_argument("--early_stopping_patience", type=int, default=0,)  #Early stopping patience（0 表示禁用）
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0,)  #Early stopping 最小改善阈值。
    p.add_argument("--save_top_k", type=int, default=3,
                   help="ModelCheckpoint 保留数：-1=保存每个验证 ckpt（长训练回头看用），>0=只留最优 K 个")

    # 5. 运行与工程参数
    p.add_argument("--devices", type=int, default=1)                   # 使用的 GPU/设备 数量（单节点内卡数）
    p.add_argument("--num_nodes", type=int, default=1)                 # 多节点 DDP 节点数（单节点保持 1）
    p.add_argument("--ddp_find_unused_parameters", action="store_true", default=False,
                   help="多卡 DDP 打开 find_unused_parameters(默认关)。纯 MLM 对照(phylo_w=protein_w=0)下 "
                        "genus_mask_token 等参数天然闲置会触发 DDP unused-parameter 报错,这时需打开;"
                        "只影响梯度同步记账,数值等价")
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "16-mixed", "32"])
    p.add_argument("--seed", type=int, default=42)                     # 随机种子，用于可复现
    p.add_argument("--accumulate_grad_batches", type=int, default=1)   # 梯度累积步数
    p.add_argument("--gradient_clip_val", type=float, default=1.0)     # 梯度裁剪阈值
    p.add_argument("--grad_checkpointing", action="store_true", default=False,
                   help="激活重算（以时间换显存，单卡可开更大 batch）；默认关，本次正式训练不用")
    p.add_argument("--num_workers", type=int, default=4)               # DataLoader 的 num_workers
    p.add_argument("--log_dir", type=str, default="tmp/logs")          # 日志保存目录
    p.add_argument(
        "--run_name",
        type=str,
        default="pretrain_stage0",
        help="日志/checkpoint 子目录名，对应 run_pretrain_once 的 log_subdir。",
    )
    p.add_argument("--no_progress_bar", action="store_true", default=False)

    # DAPT 续训:从已有 ckpt 加载初始权重(非 trainer resume,只 init state_dict)
    p.add_argument(
        "--init_from_ckpt", type=str, default=None,
        help="从该 ckpt 加载初始 state_dict;buffer 缺失会自动跳过(strict=False)。用于 DAPT 二阶段。",
    )

    return p


def _args_to_config(args: argparse.Namespace) -> PretrainRunConfig:
    return PretrainRunConfig(
        h5ad_path=args.h5ad_path,
        bias_type=args.bias_type,
        phylo_mlp_hidden=args.phylo_mlp_hidden,
        phylo_bias_last_layer_bias=not args.phylo_bias_no_last_bias,
        tree_loss_weight=args.tree_loss_weight,
        tree_n_pairs=args.tree_n_pairs,
        tree_n_triplets=args.tree_n_triplets,
        tree_margin=args.tree_margin,
        # X2 多任务(2026-05-28 夜)
        mlm_weight=args.mlm_weight,
        x2_phylo_weight=args.x2_phylo_weight,
        x2_protein_weight=args.x2_protein_weight,
        x2_head_hidden=args.x2_head_hidden,
        use_protein_pe=args.use_protein_pe,
        protein_pe_hidden=args.protein_pe_hidden,
        # Phylo Soft-Target CE(2026-05-29)
        phylo_ce_weight=args.phylo_ce_weight,
        phylo_ce_tau=args.phylo_ce_tau,
        # Phylo Tree-Wasserstein simplified(2026-05-29 phase 2)
        phylo_w_weight=args.phylo_w_weight,
        # Protein Tree-Wasserstein simplified(2026-05-30,phylo_w 镜像)
        protein_w_weight=args.protein_w_weight,
        protein_pe_path=args.protein_pe_path,
        protein_dist_path=args.protein_dist_path,
        # 对比学习(2026-06-04,InfoNCE)
        contrastive_weight=args.contrastive_weight,
        contrastive_temp=args.contrastive_temp,
        contrastive_proj_dim=args.contrastive_proj_dim,
        contrastive_mask_prob=args.contrastive_mask_prob,
        # JEPA(2026-06-04,潜空间预测)
        jepa_weight=args.jepa_weight,
        jepa_mask_ratio=args.jepa_mask_ratio,
        jepa_mlm_mask_prob=args.jepa_mlm_mask_prob,
        jepa_ema_decay=args.jepa_ema_decay,
        jepa_pred_dim=args.jepa_pred_dim,
        jepa_pred_depth=args.jepa_pred_depth,
        jepa_pred_heads=args.jepa_pred_heads,
        jepa_vicreg_weight=args.jepa_vicreg_weight,
        jepa_mask_mode=args.jepa_mask_mode,
        jepa_addr_mode=args.jepa_addr_mode,
        jepa_n_seeds=args.jepa_n_seeds,
        # JEPA v2(2026-06-06)
        jepa_global_weight=args.jepa_global_weight,
        jepa_n_reg_tokens=args.jepa_n_reg_tokens,
        jepa_ratio_start=args.jepa_ratio_start,
        jepa_ratio_end=args.jepa_ratio_end,
        # JEPA v3(2026-06-11,set 级)
        jepa_setlevel=args.jepa_setlevel,
        jepa_loss_type=args.jepa_loss_type,
        jepa_ema_end=args.jepa_ema_end,
        jepa_ema_warmup_steps=args.jepa_ema_warmup_steps,
        jepa_student_vicreg_weight=args.jepa_student_vicreg_weight,
        jepa_predict_residual=args.jepa_predict_residual,
        # 去批次(默认关)
        study_balanced=args.study_balanced,
        use_study_conditioning=args.use_study_conditioning,
        study_min_size=args.study_min_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        ff_ratio=args.ff_ratio,
        num_abundance_bins=args.num_abundance_bins,
        abundance_mode=args.abundance_mode,
        min_abundance=args.min_abundance,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        mask_prob=args.mask_prob,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_plateau_factor=args.lr_plateau_factor,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_min_lr=args.lr_plateau_min_lr,
        budget_mode=args.budget_mode,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        val_interval_epochs=args.val_interval_epochs,
        val_interval_steps=args.val_interval_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        devices=args.devices,
        num_nodes=args.num_nodes,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        precision=args.precision,
        seed=args.seed,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        grad_checkpointing=args.grad_checkpointing,
        num_workers=args.num_workers,
        log_dir=args.log_dir,
        no_progress_bar=args.no_progress_bar,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        save_top_k=args.save_top_k,
        # V5
        abundance_encoding=args.abundance_encoding,
        abundance_value_transform=args.abundance_value_transform,
        abundance_loss=args.abundance_loss,
        use_phylo_pe=not args.no_phylo_pe,
        phylo_pe_hidden=args.phylo_pe_hidden,
        pooling_mode=args.pooling_mode,
        pma_nhead=args.pma_nhead,
        pma_k=args.pma_k,
        use_metadata_task=not args.no_metadata_task,
        metadata_loss_weight=args.metadata_loss_weight,
        huber_beta=args.huber_beta,
        init_from_ckpt=args.init_from_ckpt,
    )


def main():
    args = build_argparser().parse_args()
    config = _args_to_config(args)

    rank_zero_info(f"{TAG} Loading train indices from {args.train_indices_path} ...")
    train_indices = np.load(args.train_indices_path)
    rank_zero_info(f"{TAG} Loading val indices from {args.val_indices_path} ...")
    val_indices = np.load(args.val_indices_path)

    run_pretrain_once(config, train_indices, val_indices, log_subdir=args.run_name)


if __name__ == "__main__":
    main()
