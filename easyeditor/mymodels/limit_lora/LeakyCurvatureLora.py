import torch
import torch.nn as nn
from peft.tuners.lora.layer import LoraLayer
from peft import LoraConfig


# =============================================================================
# LeakyCurvatureLora —— 带泄漏率的曲率感知 LoRA Variant
# =============================================================================
# 核心思想：
#   KFac 协方差矩阵的 **高曲率方向**（对应大特征值的特征向量 U_bar）
#   代表预训练知识中"变化敏感、容易被破坏"的方向。
#   普通 LoRA 允许梯度在这些方向上自由更新，容易引发遗忘。
#
#   本 Variant 在 **输入侧** 和 **输出侧** 各插入一个"软投影"：
#       x_proj  = x  - (1 - leak) * 高曲率输入分量
#       h_proj  = h  - (1 - leak) * 高曲率输出分量
#   当 leak → 0 时退化为完全屏蔽高曲率方向（硬投影）；
#   当 leak → 1 时退化为普通 LoRA（完全放通）。
#   leak 本身是一个可学习参数，训练过程中自动寻找平衡点。
#
# 数据流总览（Forward）：
#
#   x  [B, T, d_in]
#    │
#    ├─── high_curv_in = (x @ U_in_bar) @ U_in_bar.T   ─── 投影到高曲率输入子空间
#    │                   [B,T,k_in]     [B,T,d_in]
#    │
#    ├─── x_proj = x - (1-leak) * high_curv_in          ─── 按泄漏率软屏蔽高曲率输入方向
#    │
#    ├─── dropout(x_proj)                                ─── lora_dropout
#    ├─── lora_A(...)    [B,T,d_in] → [B,T,r]           ─── 低秩下投影
#    ├─── lora_B(...)    [B,T,r]    → [B,T,d_out]        ─── 低秩上投影
#    │    h = lora_B(lora_A(dropout(x_proj)))
#    │
#    ├─── high_curv_out = (h @ U_out_bar) @ U_out_bar.T ─── 投影到高曲率输出子空间
#    │                    [B,T,k_out]    [B,T,d_out]
#    │
#    ├─── h_proj = h - (1-leak) * high_curv_out          ─── 按泄漏率软屏蔽高曲率输出方向
#    │
#    └─── result + h_proj * scaling                      ─── 残差叠加，返回最终输出
#
# 形状约定：
#   d_in   = module.in_features
#   d_out  = module.out_features
#   r      = lora_rank（低秩瓶颈维度）
#   k_in   = 高曲率输入方向数量（由能量阈值决定）
#   k_out  = 高曲率输出方向数量
# =============================================================================

class LeakyCurvatureLora(LoraConfig):

    @staticmethod
    def init(module: LoraLayer, adapter_name: str, **kwargs) -> None:
        base_weight = module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        module.register_buffer(f"U_in_bar_{adapter_name}", base_weight.new_zeros((module.in_features, 0)))
        module.register_buffer(f"U_out_bar_{adapter_name}", base_weight.new_zeros((module.out_features, 0)))
        module.register_parameter(
            f"leak_rate_{adapter_name}",
            nn.Parameter(torch.tensor([-4.0], dtype=base_weight.dtype))
        )

    @staticmethod
    def forward(
        module: LoraLayer, active_adapter: str, x: torch.Tensor, result: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        U_in_bar  = getattr(module, f"U_in_bar_{active_adapter}").to(dtype=x.dtype)
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}").to(dtype=x.dtype)


        raw_leak = getattr(module, f"leak_rate_{active_adapter}")
        leak = torch.sigmoid(raw_leak) * 0.2   # todo: 之后设置为超参数，泄漏率上界

        lora_A  = module.lora_A[active_adapter]      
        lora_B  = module.lora_B[active_adapter]     
        dropout = module.lora_dropout[active_adapter]    
        scaling = module.scaling[active_adapter]        

        high_curv_in = (x @ U_in_bar) @ U_in_bar.T
        x_proj = x - (1.0 - leak) * high_curv_in       

        h = lora_B(lora_A(dropout(x_proj)))            

        high_curv_out = (h @ U_out_bar) @ U_out_bar.T
        h_proj = h - (1.0 - leak) * high_curv_out    

        return result + h_proj * scaling       

    @staticmethod
    def _compute_delta_weight(module: LoraLayer, active_adapter: str) -> torch.Tensor:
        # ── 取投影基与 LoRA 权重 ──────────────────────────────────────────────
        U_in_bar  = getattr(module, f"U_in_bar_{active_adapter}")   # (d_in,  k_in)
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}")  # (d_out, k_out)
        weight_A  = module.lora_A[active_adapter].weight            # (r,     d_in)
        weight_B  = module.lora_B[active_adapter].weight            # (d_out, r)
        scaling   = module.scaling[active_adapter]                  # scalar

        # 统一设备和 dtype（以 weight_B 为基准）
        device = weight_B.device
        dtype  = weight_B.dtype

        # leak: scalar ∈ (0, 0.2)，与 forward 中保持一致
        leak = torch.sigmoid(getattr(module, f"leak_rate_{active_adapter}").to(device=device)) * 0.2

        # U_in_bar / U_out_bar 迁移到同设备/dtype
        U_in_bar  = U_in_bar.to(device=device, dtype=dtype)    # (d_in,  k_in)
        U_out_bar = U_out_bar.to(device=device, dtype=dtype)   # (d_out, k_out)

        # ── LoRA 原始增量 ──────────────────────────────────────────────────────
        # BA: (d_out, d_in)
        BA = weight_B @ weight_A

        # ── 输出侧软投影：P_out @ BA ───────────────────────────────────────────
        # U_out_bar.T @ BA          : (k_out, d_in)
        # U_out_bar @ (...)         : (d_out, d_in)
        # P_out_BA = I_proj @ BA = BA - (1-leak) * U_out U_out^T @ BA
        P_out_BA = BA - (1.0 - leak) * U_out_bar @ (U_out_bar.T @ BA)  # (d_out, d_in)

        # ── 输入侧软投影：P_out_BA @ P_in ─────────────────────────────────────
        # P_out_BA @ U_in_bar       : (d_out, k_in)
        # (P_out_BA @ U_in_bar) @ U_in_bar.T : (d_out, d_in)
        # delta = P_out_BA @ (I - (1-leak) * U_in U_in^T)
        delta = P_out_BA - (1.0 - leak) * (P_out_BA @ U_in_bar) @ U_in_bar.T  # (d_out, d_in)

        # ── 乘以缩放系数返回 ────────────────────────────────────────────────────
        return delta * scaling   # (d_out, d_in)
