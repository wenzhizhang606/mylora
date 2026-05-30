"""
mymodels：自定义 LoRA 方案集合
=====================================
包含三大类方案：
  - limit_grad_lora  : 限制梯度方向的 LoRA（SchemeA/C/D）
  - limit_param_lora : 限制参数增量本身的 LoRA（CurvatureLora）
  - safe_lora        : SAFE-LoRA 谱自适应特征增强 LoRA（新一代）

公共超参数：
  - CrispLoRAHyperParams : 统一的超参数数据类
  - SAFELoRAHyperParams  : SAFE-LoRA 专用超参数
"""

# 公共超参数
from .hparams import CrispLoRAHyperParams

# ── 限制梯度方向的 LoRA ──────────────────────────────────────────────────────
# 方案A：KFac 边缘化投影优化器 + 模型包装
from .limit_grad_lora import (
    ProjectedLoRAOptimizer,
    build_lora_projection_cache,
    map_proj_cache_to_lora_params,
    wrap_model_and_build_projected_optimizer,
    apply_limit_grad_lora_to_model,
    MyLoRAHyperParams,
    compute_marginal_masks
)

# ── 限制参数增量的 LoRA ──────────────────────────────────────────────────────
# CurvatureLora：通过投影子空间约束 LoRA 增量本身
from .limit_param_lora import (
    CurvatureLora,
    attach_curvature_lora_variant
)

from .limit_lora import(
  apply_leaky_lora_to_model
)

from .lora_proj.utils import apply_lora_to_model

# ── SAFE-LoRA：谱自适应特征增强 LoRA ─────────────────────────────────────────
from .safe_lora import (
    SAFELoRAHyperParams,
    apply_safe_lora_to_model,
    build_safe_lora_projection_cache,
    project_delta_w_and_rewrite,
    project_all_lora_layers,
    compute_leak_upper_bound,
    compute_spectrum_sensitivity,
    compute_safety_scores,
    compute_danger_vector,
    compute_attenuation_from_danger,
)

from .finetune import(
    apply_simple_finetune,
)
__all__ = [
    # 超参数
    "CrispLoRAHyperParams",
    "SAFELoRAHyperParams",
    "MyLoRAHyperParams",
    # 方案A
    "ProjectedLoRAOptimizer",
    "build_lora_projection_cache",
    "map_proj_cache_to_lora_params",
    "wrap_model_and_build_projected_optimizer",
    "apply_limit_grad_lora_to_model",
    "compute_marginal_masks",
    # 调整参数增量
    "CurvatureLora",
    "attach_curvature_lora_variant",
    "apply_lora_to_model",
    # SAFE-LoRA
    "apply_safe_lora_to_model",
    "build_safe_lora_projection_cache",
    "project_delta_w_and_rewrite",
    "project_all_lora_layers",
    "compute_leak_upper_bound",
    "compute_spectrum_sensitivity",
    "compute_safety_scores",
    "compute_danger_vector",
    "compute_attenuation_from_danger",

    "apply_simple_finetune",
]
