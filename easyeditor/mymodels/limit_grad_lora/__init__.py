# 核心组件：KFac 边缘化投影优化器（带泄漏率软投影）
from .projected_lora_optimizer import ProjectedLoRAOptimizer

# 工具函数：KFac 统计、投影缓存构建、leak_rate 注册、模型包装
from .utils import (
    build_lora_projection_cache,
    map_proj_cache_to_lora_params,
    wrap_model_and_build_projected_optimizer,
    apply_limit_grad_lora_to_model,
    compute_marginal_masks,
    get_topk_indices_by_energy_ratio,
    _register_leak_rate_for_layer,
)


__all__ = [
    "ProjectedLoRAOptimizer",
    "build_lora_projection_cache",
    "map_proj_cache_to_lora_params",
    "wrap_model_and_build_projected_optimizer",
    "apply_limit_grad_lora_to_model",
    "compute_marginal_masks",
    "get_topk_indices_by_energy_ratio",
    "_register_leak_rate_for_layer",
]
