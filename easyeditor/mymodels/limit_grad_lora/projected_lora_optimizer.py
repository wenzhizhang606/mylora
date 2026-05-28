import torch
from torch.optim import Adam
from typing import Dict, Optional


class ProjectedLoRAOptimizer(Adam):
    def __init__(
        self,
        params,
        projection_cache_map: Dict,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        projection_mode: str = "marginal_AB",
        use_leak:bool=False,
        leak_rate:float =0.0,
        newton_damping: float = 1e-3,
        use_dynamic_projection: bool = True,
        dynamic_projection_beta: float = 0.95,
        dynamic_projection_strength: float = 0.5,
        dynamic_projection_min_scale: float = 0.2,
    ):
        defaults = dict(
            projection_cache_map={},
            projection_mode=projection_mode,
        )
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
        )
        for group in self.param_groups:
            group.update(defaults)

        self._preload_cache(projection_cache_map)
        self.use_leak = use_leak
        self.leak_rate = leak_rate
        self.newton_damping = max(float(newton_damping), 0.0)
        self.use_dynamic_projection = use_dynamic_projection
        self.dynamic_projection_beta = min(max(float(dynamic_projection_beta), 0.0), 0.9999)
        self.dynamic_projection_strength = min(max(float(dynamic_projection_strength), 0.0), 1.0)
        self.dynamic_projection_min_scale = min(
            max(float(dynamic_projection_min_scale), 0.0), 1.0
        )


    def _preload_cache(self, projection_cache_map: Dict):
        """
        将投影矩阵（Ua/Ub/mask_a/mask_b）提前搬到对应参数所在设备。
        leak_rate_param 是 nn.Parameter，直接保留引用，无需额外迁移。

        构建后写回每个 param_group["projection_cache_map"]。
        """
        # 建立 id(param) → param 的索引，用于跨 group 匹配
        all_params: Dict[int, torch.nn.Parameter] = {}
        for group in self.param_groups:
            for p in group["params"]:
                all_params[id(p)] = p

        preloaded: Dict[torch.nn.Parameter, Dict] = {}
        for param, cache in projection_cache_map.items():
            if id(param) not in all_params:
                continue
            p   = all_params[id(param)]
            dev = p.device
            dtype = p.dtype

            new_cache: Dict = {"param_type": cache.get("param_type", "unknown")}

            # 投影矩阵：搬到 GPU 并转为参数同 dtype
            if "Ua" in cache:
                new_cache["Ua"]     = cache["Ua"].to(device=dev, dtype=dtype)
            if "mask_a" in cache:
                new_cache["mask_a"] = cache["mask_a"].to(device=dev, dtype=dtype)
            if "eig_a" in cache:
                new_cache["eig_a"] = cache["eig_a"].to(device=dev, dtype=dtype)
            if "Ub" in cache:
                new_cache["Ub"]     = cache["Ub"].to(device=dev, dtype=dtype)
            if "mask_b" in cache:
                new_cache["mask_b"] = cache["mask_b"].to(device=dev, dtype=dtype)
            if "eig_b" in cache:
                new_cache["eig_b"] = cache["eig_b"].to(device=dev, dtype=dtype)

            # leak_rate_param：直接保留 nn.Parameter 引用（已在正确设备上）
            new_cache["leak_rate_param"] = cache.get("leak_rate_param", None)

            preloaded[param] = new_cache

        for group in self.param_groups:
            group["projection_cache_map"] = preloaded

        print(f"[ProjectedLoRAOptimizer] 已预加载 {len(preloaded)} 个参数的投影矩阵到 GPU")

    def reset_cache(self, new_projection_cache_map: Dict):
        """
        更新投影缓存，并将旧缓存对应的动量缓冲区同步到新子空间。
        连续编辑场景下，每轮编辑前调用此方法。

        流程：
          1. 用旧 cache（已在 GPU）对已有动量做软投影对齐
          2. 重新预加载新 cache
        """
        # step 1：动量缓冲区同步到旧子空间（防止旧动量污染新方向）
        for group in self.param_groups:
            old_cache_map = group.get("projection_cache_map", {})
            mode = group.get("projection_mode", "marginal_AB")
            for p in group["params"]:
                if p not in self.state or p not in old_cache_map:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    continue
                cache = old_cache_map[p]
                param_type = cache.get("param_type", "unknown")
                m = state["exp_avg"]
                m_proj = self._project_grad(m, cache, param_type, mode)
                if m_proj is not None:
                    m.copy_(m_proj)

        # step 2：重新预加载新 cache
        self._preload_cache(new_projection_cache_map)

    # ──────────────────────────────────────────────────────────────────────────
    # 软投影核心
    # ──────────────────────────────────────────────────────────────────────────

    def _project_grad(
        self,
        grad: torch.Tensor,
        cache: Dict,
        param_type: str,
        mode: str,
        param: Optional[torch.nn.Parameter] = None,
        adapt_projection: bool = False,
    ) -> Optional[torch.Tensor]:
        # ── 读取泄漏率（detach：仅读值，不影响计算图）──────────────────────────
        leak_rate_param = cache.get("leak_rate_param", None)
        # 如果没有泄露率，则退回到第一个版本
        if self.use_leak:
            leak = torch.sigmoid(leak_rate_param.detach().to(device=grad.device, dtype=grad.dtype)) * self.leak_rate
        else:
            leak = torch.zeros(1, device=grad.device, dtype=grad.dtype)

        def _newton_strength(eigvals: torch.Tensor) -> torch.Tensor:
            eigvals = torch.clamp(eigvals.to(device=grad.device, dtype=grad.dtype), min=0.0)
            if eigvals.numel() == 0:
                return eigvals
            damping_scale = self.newton_damping * eigvals.max().clamp(min=1e-12)
            return (1.0 - leak) * eigvals / (eigvals + damping_scale)

        def _apply_dynamic_projection(
            base_strength: torch.Tensor,
            direction_coeffs: torch.Tensor,
            reduce_dim: int,
            state_key: str,
        ) -> torch.Tensor:
            # Continual-learning style task adaptation:
            # if the current downstream task repeatedly uses a protected
            # direction, its EMA score grows and we temporarily relax deletion
            # on that direction. The K-FAC/Newton subspace is still the anchor;
            # only the diagonal deletion strength changes online.
            if (
                not self.use_dynamic_projection
                or param is None
                or base_strength.numel() == 0
                or self.dynamic_projection_strength <= 0
            ):
                return base_strength

            direction_energy = (
                direction_coeffs.detach()
                .to(dtype=torch.float32)
                .pow(2)
                .mean(dim=reduce_dim)
            )
            state = self.state[param]
            ema = state.get(state_key, None)
            if ema is None or ema.shape != direction_energy.shape:
                ema = torch.zeros_like(direction_energy)

            if adapt_projection:
                beta = self.dynamic_projection_beta
                ema.mul_(beta).add_(direction_energy, alpha=1.0 - beta)
                state[state_key] = ema

            task_score = ema / ema.max().clamp(min=1e-12)
            keep_scale = 1.0 - self.dynamic_projection_strength * task_score
            keep_scale = keep_scale.clamp(
                min=self.dynamic_projection_min_scale,
                max=1.0,
            )
            return base_strength * keep_scale.to(device=grad.device, dtype=grad.dtype)

        if param_type == "lora_A":
            # lora_A: (r, d_in)，对输入方向做右投影
            if mode not in ("marginal_A", "marginal_AB"):
                return None
            if "Ua" not in cache or "mask_a" not in cache:
                return None

            mask_a = cache["mask_a"]   # (d_in, k_in)，列为高曲率特征向量
            eig_a = cache.get("eig_a", None)

            if eig_a is None:
                grad_high = grad @ (mask_a @ mask_a.T)
                grad_proj = grad - (1.0 - leak) * grad_high
                return grad_proj

            direction_coeffs = grad @ mask_a
            delete_strength = _newton_strength(eig_a)
            delete_strength = _apply_dynamic_projection(
                delete_strength,
                direction_coeffs,
                reduce_dim=0,
                state_key="dynamic_projection_ema_a",
            )
            grad_high = (direction_coeffs * delete_strength.unsqueeze(0)) @ mask_a.T
            grad_proj = grad - grad_high
            return grad_proj

        elif param_type == "lora_B":
            # lora_B: (d_out, r)，对输出方向做左投影
            if mode not in ("marginal_B", "marginal_AB"):
                return None
            if "Ub" not in cache or "mask_b" not in cache:
                return None

            mask_b = cache["mask_b"]   # (d_out, k_out)，列为高曲率特征向量
            eig_b = cache.get("eig_b", None)

            if eig_b is None:
                grad_high = (mask_b @ mask_b.T) @ grad
                grad_proj = grad - (1.0 - leak) * grad_high
                return grad_proj

            # grad: (d_out, r)，左乘带 Newton 权重的投影矩阵
            direction_coeffs = mask_b.T @ grad
            delete_strength = _newton_strength(eig_b)
            delete_strength = _apply_dynamic_projection(
                delete_strength,
                direction_coeffs,
                reduce_dim=1,
                state_key="dynamic_projection_ema_b",
            )
            grad_high = mask_b @ (delete_strength.unsqueeze(-1) * direction_coeffs)
            grad_proj = grad - grad_high
            return grad_proj

        else:
            return None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            cache_map = group.get("projection_cache_map", {})
            mode      = group.get("projection_mode", "marginal_AB")
            if not cache_map:
                print("[ProjectedLoRAOptimizer] 警告：cache_map 为空，跳过梯度投影")
                continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p not in cache_map:
                    continue

                cache      = cache_map[p]
                param_type = cache.get("param_type", "unknown")

                # 软投影梯度
                grad_proj = self._project_grad(
                    p.grad,
                    cache,
                    param_type,
                    mode,
                    param=p,
                    adapt_projection=True,
                )
                if grad_proj is not None:
                    p.grad.copy_(grad_proj)

        return super().step(closure)
