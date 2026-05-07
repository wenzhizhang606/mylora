"""
SAFE-LoRA: Spectrum-Adaptive Feature-Enriched LoRA
====================================================
安全微调的改进方法，包含三层创新：

  层次一：连续谱自适应正则化
    用特征值驱动的连续衰减替代二元 mask 分类，
    消除 energy_threshold 超参数，平滑方向间过渡。

  层次二：对比安全方向发现
    同时计算预训练 K-FAC 和安全数据 K-FAC，
    识别高曲率但对安全任务关键的方向，避免误伤。

  层次三：ΔW 级联合投影 + 渐进约束收紧
    在每次 optimizer step 后对完整 ΔW = B @ A 做谱投影，
    并通过余弦退火从 leak_max 收紧到 leak_min。
"""

from .SAFELoRA_hparams import SAFELoRAHyperParams
from .utils import (
    # 主训练入口
    apply_safe_lora_to_model,

    # 投影缓存构建
    build_safe_lora_projection_cache,

    # ΔW 投影
    project_delta_w_and_rewrite,
    project_all_lora_layers,

    # 渐进约束
    compute_leak_upper_bound,

    # 谱分析（可单独使用）
    compute_spectrum_sensitivity,
    compute_safety_scores,
    compute_danger_vector,
    compute_attenuation_from_danger,
)

__all__ = [
    # 超参数
    "SAFELoRAHyperParams",

    # 主入口
    "apply_safe_lora_to_model",

    # 缓存构建
    "build_safe_lora_projection_cache",

    # ΔW 投影
    "project_delta_w_and_rewrite",
    "project_all_lora_layers",

    # 渐进约束
    "compute_leak_upper_bound",

    # 谱分析
    "compute_spectrum_sensitivity",
    "compute_safety_scores",
    "compute_danger_vector",
    "compute_attenuation_from_danger",
]
