from typing import Dict, Optional, Tuple

import torch
from torch.optim import Adam


class ProjectedAdam(Adam):
    """
    第 3.2 节 K-FAC 软约束更新对应的 Adam 包装器。

    对于一个权重块 W 及其梯度 G，K-FAC 软约束 Newton 系统为

        B_e dW A_e + lambda B_c dW A_c = -G.

    本实现会构造广义基 Q_A、Q_B，使其满足

        Q_A.T A_e Q_A = I, Q_A.T A_c Q_A = diag(a)
        Q_B.T B_e Q_B = I, Q_B.T B_c Q_B = diag(b)

    然后用下式替换原始梯度

        Q_B [(Q_B.T G Q_A) / (1 + lambda * outer(b, a))] Q_A.T.

    随后 Adam 执行常规下降步骤，因此参数更新方向等价于该预条件梯度的负方向。
    """

    _PRECOMPUTED_BASIS_KEYS = (
        ("soft_q_a", "soft_q_b", "soft_eig_a", "soft_eig_b"),
    )

    _FACTOR_KEY_SETS = (
        (("edit_A", "edit_B"), ("cap_A", "cap_B")),
    )

    def __init__(
        self,
        params,
        projection_cache_map: Optional[Dict] = None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        additional_projection_cache_map: Optional[Dict] = None,
        soft_lambda: float = 1.0,
        factor_damping: float = 1e-5,
        cache_generalized_basis: bool = True,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        defaults = {
            "projection_cache_map": projection_cache_map or {},
            "additional_projection_cache_map": additional_projection_cache_map or {},
            "soft_lambda": float(soft_lambda),
        }
        for group in self.param_groups:
            group.update(defaults)

        self.factor_damping = max(float(factor_damping), 0.0)
        self.cache_generalized_basis = bool(cache_generalized_basis)

    def reset_cache_old(self, new_projection_cache_map):
        self.reset_cache(new_projection_cache_map)

    def reset_cache(self, new_projection_cache_map):
        new_projection_cache_map = new_projection_cache_map or {}
        for group in self.param_groups:
            group["projection_cache_map"] = new_projection_cache_map
            self._project_momentum(group, new_projection_cache_map)

    def reset_additional_cache(self, additional_projection_cache_map):
        additional_projection_cache_map = additional_projection_cache_map or {}
        for group in self.param_groups:
            group["additional_projection_cache_map"] = additional_projection_cache_map
            self._project_momentum(group, additional_projection_cache_map)

    def _project_momentum(self, group, cache_map: Dict):
        for p in group["params"]:
            if p not in self.state or p not in cache_map:
                continue

            state = self.state[p]
            exp_avg = state.get("exp_avg", None)
            if exp_avg is None or exp_avg.ndim != 2:
                continue

            projected = self._soft_kfac_precondition(
                exp_avg,
                cache_map[p],
                soft_lambda=group.get("soft_lambda", 1.0),
            )
            if projected is not None:
                exp_avg.copy_(projected)

    @staticmethod
    def _tensor(cache: Dict, key: str, like: torch.Tensor, dtype: torch.dtype):
        value = cache.get(key, None)
        if value is None:
            return None
        return value.to(device=like.device, dtype=dtype)

    @staticmethod
    def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _check_square(matrix: torch.Tensor, name: str):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a square matrix, got {tuple(matrix.shape)}.")

    @staticmethod
    def _check_basis_shape(
        tensor: torch.Tensor,
        q_a: torch.Tensor,
        q_b: torch.Tensor,
        eig_a: torch.Tensor,
        eig_b: torch.Tensor,
    ):
        out_dim, in_dim = tensor.shape
        if q_a.ndim != 2 or q_b.ndim != 2:
            raise ValueError(
                f"Generalized bases must be matrices, got q_a={tuple(q_a.shape)}, "
                f"q_b={tuple(q_b.shape)}."
            )
        if q_a.shape[0] != in_dim or q_b.shape[0] != out_dim:
            raise ValueError(
                "Generalized basis dimension mismatch: "
                f"tensor={tuple(tensor.shape)}, q_a={tuple(q_a.shape)}, "
                f"q_b={tuple(q_b.shape)}."
            )
        if eig_a.numel() != q_a.shape[1] or eig_b.numel() != q_b.shape[1]:
            raise ValueError(
                "Generalized eigenvalue dimension mismatch: "
                f"eig_a={tuple(eig_a.shape)}, q_a={tuple(q_a.shape)}, "
                f"eig_b={tuple(eig_b.shape)}, q_b={tuple(q_b.shape)}."
            )

    def _generalized_basis(
        self,
        edit_factor: torch.Tensor,
        cap_factor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_square(edit_factor, "edit_factor")
        self._check_square(cap_factor, "cap_factor")
        if edit_factor.shape != cap_factor.shape:
            raise ValueError(
                "K-FAC factor shape mismatch: "
                f"edit={tuple(edit_factor.shape)}, cap={tuple(cap_factor.shape)}."
            )

        edit_factor = self._symmetrize(edit_factor)
        cap_factor = self._symmetrize(cap_factor)

        edit_eigs, edit_vecs = torch.linalg.eigh(edit_factor)
        scale = edit_eigs.abs().max().clamp(min=1.0)
        floor = self.factor_damping * scale
        edit_eigs = torch.clamp(edit_eigs, min=floor)

        inv_sqrt = edit_vecs @ torch.diag(torch.rsqrt(edit_eigs)) @ edit_vecs.T
        whitened_cap = self._symmetrize(inv_sqrt @ cap_factor @ inv_sqrt)
        cap_eigs, cap_vecs = torch.linalg.eigh(whitened_cap)

        q = inv_sqrt @ cap_vecs
        cap_eigs = torch.clamp(cap_eigs, min=0.0)
        return q.contiguous(), cap_eigs.contiguous()

    def _basis_from_precomputed(
        self,
        tensor: torch.Tensor,
        cache: Dict,
        dtype: torch.dtype,
    ):
        for q_a_key, q_b_key, eig_a_key, eig_b_key in self._PRECOMPUTED_BASIS_KEYS:
            if all(key in cache for key in (q_a_key, q_b_key, eig_a_key, eig_b_key)):
                q_a = self._tensor(cache, q_a_key, tensor, dtype)
                q_b = self._tensor(cache, q_b_key, tensor, dtype)
                eig_a = self._tensor(cache, eig_a_key, tensor, dtype)
                eig_b = self._tensor(cache, eig_b_key, tensor, dtype)
                return q_a, q_b, eig_a.flatten(), eig_b.flatten()
        return None

    def _factors_from_cache(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        for edit_keys, cap_keys in self._FACTOR_KEY_SETS:
            if all(key in cache for key in (*edit_keys, *cap_keys)):
                edit_a = self._tensor(cache, edit_keys[0], tensor, dtype)
                edit_b = self._tensor(cache, edit_keys[1], tensor, dtype)
                cap_a = self._tensor(cache, cap_keys[0], tensor, dtype)
                cap_b = self._tensor(cache, cap_keys[1], tensor, dtype)
                return edit_a, edit_b, cap_a, cap_b
        return None

    def _basis_from_factors(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        factors = self._factors_from_cache(tensor, cache, dtype)
        if factors is None:
            return None

        edit_a, edit_b, cap_a, cap_b = factors
        q_a, eig_a = self._generalized_basis(edit_a, cap_a)
        q_b, eig_b = self._generalized_basis(edit_b, cap_b)

        if self.cache_generalized_basis:
            cache["soft_q_a"] = q_a.detach().cpu()
            cache["soft_q_b"] = q_b.detach().cpu()
            cache["soft_eig_a"] = eig_a.detach().cpu()
            cache["soft_eig_b"] = eig_b.detach().cpu()

        return q_a, q_b, eig_a, eig_b

    def _get_generalized_basis(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        precomputed = self._basis_from_precomputed(tensor, cache, dtype)
        if precomputed is not None:
            return precomputed
        return self._basis_from_factors(tensor, cache, dtype)

    def _soft_kfac_precondition(
        self,
        tensor: torch.Tensor,
        cache: Optional[Dict],
        soft_lambda: float,
    ):
        if cache is None or tensor.ndim != 2:
            return None

        compute_dtype = (
            torch.float32
            if tensor.dtype in (torch.float16, torch.bfloat16)
            else tensor.dtype
        )
        source = tensor.to(dtype=compute_dtype)

        basis = self._get_generalized_basis(source, cache, compute_dtype)
        if basis is None:
            return None

        q_a, q_b, eig_a, eig_b = basis
        self._check_basis_shape(source, q_a, q_b, eig_a, eig_b)

        coeffs = q_b.T @ source @ q_a
        joint_eigs = torch.outer(
            torch.clamp(eig_b.flatten(), min=0.0),
            torch.clamp(eig_a.flatten(), min=0.0),
        ).to(device=source.device, dtype=source.dtype)
        denom = 1.0 + float(soft_lambda) * joint_eigs
        preconditioned = q_b @ (coeffs / denom.clamp(min=1e-12)) @ q_a.T
        return preconditioned.to(dtype=tensor.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        self._debug_step += 1
        loss = None
        if closure is not None:
            # Adam 保持可选 closure 约定：如果调用方提供 closure，
            # 就在启用梯度的情况下重新计算 loss。
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            # 主缓存是每个参数对应的主要 K-FAC 软约束缓存。
            # 按 PDF 中的记号，它应包含 H_e 和 H_c 因子，
            # 或者包含预先计算好的广义基。
            cache_map = group.get("projection_cache_map", {}) or {}

            # 额外缓存会在主缓存之后应用。
            # 当第二个保护来源需要进一步预条件化梯度时使用它。
            additional_cache_map = group.get("additional_projection_cache_map", {}) or {}

            # soft_lambda 对应 1 / (1 + lambda * b_i * a_j) 中的 lambda。
            # 该值越大，高能力曲率方向上的梯度收缩越强。
            soft_lambda = group.get("soft_lambda", 1.0)

            debug_samples = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    # 只有矩阵权重才有 K-FAC A/B 因子。
                    # bias、标量以及没有梯度的参数会退回普通 Adam 行为。
                    continue

                grad_norm = None
                if len(debug_samples) < 3:
                    grad_norm = p.grad.detach().float().norm().item()

                grad_proj = None
                if p in cache_map:
                    # 第一分支：使用主缓存执行第 3.2 节的 K-FAC 软求解。
                    # 如果缓存不完整，辅助函数会返回 None，
                    # 此处不会修改该参数的梯度。
                    grad_proj = self._soft_kfac_precondition(
                        p.grad,
                        cache_map[p],
                        soft_lambda=soft_lambda,
                    )

                # 当前不会进入
                if p in additional_cache_map:
                    # 第二分支：可选地再执行一次软求解。
                    # 如果主分支已经生成梯度，就基于该结果继续；
                    # 否则从原始梯度开始。
                    source_grad = grad_proj if grad_proj is not None else p.grad
                    additional_proj = self._soft_kfac_precondition(
                        source_grad,
                        additional_cache_map[p],
                        soft_lambda=soft_lambda,
                    )
                    if additional_proj is not None:
                        # 只有额外缓存可用时才覆盖结果。
                        # 如果第二个缓存缺少必要因子，就保留主分支结果。
                        grad_proj = additional_proj

                if grad_proj is not None:
                    if len(debug_samples) < 3:
                        if grad_norm is None:
                            grad_norm = p.grad.detach().float().norm().item()
                        proj_norm = grad_proj.detach().float().norm().item()
                        debug_samples.append(
                            f"shape={tuple(p.shape)}, "
                            f"grad_norm={grad_norm:.4e}, "
                            f"proj_norm={proj_norm:.4e}"
                        )
                    # Adam 会在 super().step() 内部读取 p.grad。
                    # 因此这里原地写回预条件化后的梯度，
                    # 让 Adam 的动量从投影/软求解后的梯度更新。
                    p.grad.copy_(grad_proj)
            for sample_idx, sample in enumerate(debug_samples):
                print(
                    "[ProjectedAdam] "
                    f"step={self._debug_step} group={group_idx} "
                    f"sample_{sample_idx}: {sample}"
                )
        return super().step(closure)
