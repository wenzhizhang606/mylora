from dataclasses import dataclass, field
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class MyLoRAHyperParams(HyperParams):
    # 基本信息
    alg_name: str
    model_name: str
    device: int
     
    # 编辑目标层
    layers: List[int]
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # LoRA 配置
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    target_modules: List[str]
    lora_type: str = "lora"

    # 训练超参数
    num_steps: int = 20
    lr: float = 5e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    max_length: int = 40
    objective_optimization: str = "target_new"

    # K-FAC 协方差统计
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 10000
    mom2_dtype: str = "float32"
    energy_threshold: float = 0.5
    base_kfac_cache_path: str = None
    task_kfac_cache_path: str = None
    task_kfac_weight: float = 0.0
    task_kfac_tag: str = "task"
    merged_kfac_cache_path: str = None

    # KFac 初始化与投影优化器
    projection_mode: str = "marginal_AB"
    projection_method: str = "param"
    use_projection: bool = False

    # 连续编辑
    recalculate_cache: bool = False
    recalculate_weight_threshold: float = 0.1
    edit_cache_style: str = "pretrain_only"
    edit_n_samples: int = 10
    disable_old_loss_check: bool = True

    # 其他
    kl_factor: float = 0.0
    norm_constraint: float = 0.0
    model_parallel: bool = False

    use_leak:bool = False
    leak_rate: float = 0.2
    newton_damping: float = 1e-3
    use_dynamic_projection: bool = True
    dynamic_projection_beta: float = 0.95
    dynamic_projection_strength: float = 0.5
    dynamic_projection_min_scale: float = 0.2
    projection_method_lora:str = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config["alg_name"] == "MyLoRA") or print(
            f"MyLoRAHyperParams cannot load from {hparams_name_or_path}, "
            f"alg_name is {config['alg_name']}"
        )
        return cls(**config)
