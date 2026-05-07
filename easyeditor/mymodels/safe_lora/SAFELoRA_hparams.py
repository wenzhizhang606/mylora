"""
SAFELoRA 超参数配置
============================================
SAFE-LoRA: Spectrum-Adaptive Feature-Enriched LoRA

三层改进架构：
  层次一：连续谱自适应正则化（替换二元 mask）
  层次二：对比安全方向发现（预训练 vs 安全数据 K-FAC）
  层次三：ΔW 级联合投影 + 渐进约束收紧
"""

from dataclasses import dataclass, field
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class SAFELoRAHyperParams(HyperParams):
    # ── 基本信息 ──────────────────────────────────────────────────────────────
    alg_name: str = "SAFELoRA"
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    device: int = 0
    model_parallel: bool = False

    # ── 编辑目标层配置 ────────────────────────────────────────────────────────
    layers: List[int] = field(default_factory=lambda: [19, 20, 21, 22, 23])
    rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj"
    layer_module_tmp: str = "model.layers.{}"
    mlp_module_tmp: str = "model.layers.{}.mlp"
    attn_module_tmp: str = "model.layers.{}.self_attn"
    ln_f_module: str = "model.norm"
    lm_head_module: str = "lm_head"

    # ── LoRA 配置 ─────────────────────────────────────────────────────────────
    lora_type: str = "lora"
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["down_proj"])

    # ── 训练超参数 ────────────────────────────────────────────────────────────
    num_steps: int = 25
    lr: float = 5e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    max_length: int = 40
    objective_optimization: str = "target_new"

    # ── 预训练 K-FAC 统计配置 ─────────────────────────────────────────────────
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 10000
    mom2_dtype: str = "float32"

    # ── SAFE-LoRA 核心参数 ────────────────────────────────────────────────────

    # 渐进约束收紧：泄漏率上界从 leak_max 退火到 leak_min
    # leak ∈ [leak_min, leak_max]
    #   leak → 1: 完全放通（无约束）
    #   leak → 0: 完全屏蔽危险方向
    leak_max: float = 0.5          # 训练初期：允许 50% 危险方向信息通过
    leak_min: float = 0.01         # 训练末期：只允许 1% 危险方向信息通过

    # 谱温度：控制特征值 -> 敏感度映射的锐度
    #   < 1.0: 更尖锐的区分（更接近二元 mask 行为）
    #   = 1.0: 线性映射（推荐默认值）
    #   > 1.0: 更平滑的过渡
    spectrum_temperature: float = 1.0

    # ── 对比安全方向发现 ─────────────────────────────────────────────────────
    # 是否启用对比分析（预训练 K-FAC vs 安全数据 K-FAC）
    use_contrastive_analysis: bool = True

    # 安全数据 K-FAC 样本数（从编辑请求中采样）
    safety_kfac_samples: int = 100

    # 安全分数的平滑因子：避免零样本时的除零问题
    safety_score_smoothing: float = 0.01

    # ── ΔW 投影频率 ──────────────────────────────────────────────────────────
    # 每 N 步对 ΔW 做一次联合投影。1 = 每步都投影（推荐）。
    # >1 可加速训练，但约束精度会降低。
    project_every_n_steps: int = 1

    # ── 其他 ─────────────────────────────────────────────────────────────────
    kl_factor: float = 0.0
    norm_constraint: bool = False
    disable_old_loss_check: bool = True

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"
        with open(hparams_name_or_path, "r") as f:
            config = yaml.safe_load(f)
            config = super().construct_float_from_scientific_notation(config)
        return cls(**config)
