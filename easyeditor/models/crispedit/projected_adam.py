import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from dotenv import load_dotenv
from torch.optim import Adam

load_dotenv()


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
        debug_factor_stats: bool = True,
        factor_stats_quantiles: Tuple[float, ...] = (
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ),
        factor_stats_sample_size: int = 0,
        factor_stats_json_path: Optional[str] = None,
        debug_grad_norm_stats: bool = True,
        grad_norm_stats_json_path: Optional[str] = None,
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
        self.debug_factor_stats = bool(debug_factor_stats)
        self.factor_stats_quantiles = tuple(
            q
            for q in (float(q) for q in factor_stats_quantiles)
            if 0.0 <= q <= 1.0
        )
        # 0 means exact quantiles over all finite values; a positive value enables
        # deterministic subsampling for very large tensors.
        self.factor_stats_sample_size = max(int(factor_stats_sample_size), 0)
        repo_root = Path(__file__).resolve().parents[3]
        self.factor_stats_json_path = Path(
            factor_stats_json_path or repo_root / "projected_adam_factor_stats.json"
        )
        self._factor_stats_recorded = set()
        self._factor_stats_records = {
            "quantiles": list(self.factor_stats_quantiles),
            "sample_size": self.factor_stats_sample_size,
            "layers": [],
        }

        # 梯度范数统计：保存路径与 utils.py 一致（STATS_DIR），文件名不同。
        self.debug_grad_norm_stats = bool(debug_grad_norm_stats)
        _stats_dir = os.getenv("STATS_DIR")
        _grad_norm_dir = Path(_stats_dir) if _stats_dir else repo_root
        self.grad_norm_stats_json_path = Path(
            grad_norm_stats_json_path or _grad_norm_dir / "projected_adam_grad_norm_stats.json"
        )
        self._grad_norm_step = 0
        self._grad_norm_records = {"steps": []}

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

    # 原实现（基于 eigh 特征分解求 inv_sqrt），保留备查：
    # def _generalized_basis(
    #     self,
    #     edit_factor: torch.Tensor,
    #     cap_factor: torch.Tensor,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     self._check_square(edit_factor, "edit_factor")
    #     self._check_square(cap_factor, "cap_factor")
    #     if edit_factor.shape != cap_factor.shape:
    #         raise ValueError(
    #             "K-FAC factor shape mismatch: "
    #             f"edit={tuple(edit_factor.shape)}, cap={tuple(cap_factor.shape)}."
    #         )
    #
    #     edit_factor = self._symmetrize(edit_factor)
    #     cap_factor = self._symmetrize(cap_factor)
    #
    #     edit_eigs, edit_vecs = torch.linalg.eigh(edit_factor)
    #     scale = edit_eigs.abs().max().clamp(min=1.0)
    #     floor = self.factor_damping * scale
    #     edit_eigs = torch.clamp(edit_eigs, min=floor)
    #
    #     inv_sqrt = edit_vecs @ torch.diag(torch.rsqrt(edit_eigs)) @ edit_vecs.T
    #     whitened_cap = self._symmetrize(inv_sqrt @ cap_factor @ inv_sqrt)
    #     cap_eigs, cap_vecs = torch.linalg.eigh(whitened_cap)
    #
    #     q = inv_sqrt @ cap_vecs
    #     cap_eigs = torch.clamp(cap_eigs, min=0.0)
    #     return q.contiguous(), cap_eigs.contiguous()

    def _generalized_basis(self, edit_factor, cap_factor):
        self._check_square(edit_factor, "edit_factor")
        self._check_square(cap_factor, "cap_factor")

        edit_factor = self._symmetrize(edit_factor)
        cap_factor = self._symmetrize(cap_factor)

        n = edit_factor.shape[0]
        trace_scale = edit_factor.diagonal().abs().mean().clamp(min=1e-12)
        eps = self.factor_damping * trace_scale
        edit_factor_reg = edit_factor + eps * torch.eye(n, device=edit_factor.device, dtype=edit_factor.dtype)

        # 条件数统计：阻尼前后对比，诊断 edit_factor 病态程度与阻尼效果
        eigvals = torch.linalg.eigvalsh(edit_factor)
        cond = eigvals.max() / eigvals.clamp(min=1e-20).min()
        eigvals_reg = torch.linalg.eigvalsh(edit_factor_reg)
        cond_reg = eigvals_reg.max() / eigvals_reg.clamp(min=1e-20).min()
        print(f"条件数(阻尼前): {cond.item():.2e}  条件数(阻尼后): {cond_reg.item():.2e}")

        # Cholesky做广义特征分解
        L = torch.linalg.cholesky(edit_factor_reg)
        # 原实现：显式构造 L_inv 再做稠密矩阵乘（保留备查）
        # L_inv = torch.linalg.solve_triangular(L, torch.eye(n, device=L.device, dtype=L.dtype), upper=False)
        #
        # whitened_cap = self._symmetrize(L_inv @ cap_factor @ L_inv.T)
        # q = L_inv.T @ cap_vecs

        # 不显式构造 L_inv，直接对 cap_factor 做两次三角回代
        tmp = torch.linalg.solve_triangular(L, cap_factor, upper=False)        # 解 L @ tmp = cap_factor
        whitened_cap = torch.linalg.solve_triangular(L, tmp.T, upper=False).T  # 解 L @ X.T = tmp.T  =>  L^{-1} cap_factor L^{-T}
        cap_eigs, cap_vecs = torch.linalg.eigh(whitened_cap)

        # q = L^{-T} @ cap_vecs，用三角回代避免 L_inv（解 L^T @ q = cap_vecs，L^T 为上三角）
        q = torch.linalg.solve_triangular(L.transpose(-1, -2), cap_vecs, upper=True)
        cap_eigs = torch.clamp(cap_eigs, min=0.0)

        if not torch.isfinite(q).all() or not torch.isfinite(cap_eigs).all():
            raise RuntimeError("generalized basis produced non-finite values, check factor conditioning")

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

    @staticmethod
    def _exact_quantiles(values: torch.Tensor, quantiles: Tuple[float, ...]) -> Dict:
        num_values = values.numel()
        if num_values == 0 or len(quantiles) == 0:
            return {}

        kth_cache = {}

        def kth_value(zero_based_idx: int) -> float:
            zero_based_idx = max(0, min(int(zero_based_idx), num_values - 1))
            if zero_based_idx not in kth_cache:
                kth_cache[zero_based_idx] = torch.kthvalue(
                    values,
                    zero_based_idx + 1,
                ).values.item()
            return kth_cache[zero_based_idx]

        result = {}
        for quantile in quantiles:
            position = float(quantile) * (num_values - 1)
            lower_idx = int(position)
            upper_idx = min(lower_idx + 1, num_values - 1)
            lower_value = kth_value(lower_idx)
            upper_value = kth_value(upper_idx)
            weight = position - lower_idx
            result[f"{quantile:.4g}"] = lower_value + (upper_value - lower_value) * weight
        return result

    def _distribution_stats(self, values: torch.Tensor) -> Dict:
        flat = values.detach().reshape(-1)
        total = flat.numel()
        if total == 0:
            return {
                "numel": 0,
                "min": None,
                "max": None,
                "quantile_source": "empty",
                "quantiles": {},
            }

        flat = flat.float()
        finite_mask = torch.isfinite(flat)
        finite_flat = flat if finite_mask.all().item() else flat[finite_mask]
        if finite_flat.numel() == 0:
            return {
                "numel": total,
                "finite_numel": 0,
                "min": None,
                "max": None,
                "quantile_source": "non_finite",
                "quantiles": {},
            }

        min_value = finite_flat.min().item()
        max_value = finite_flat.max().item()

        quantile_source = "exact"
        quantile_values = finite_flat
        if (
            self.factor_stats_sample_size > 0
            and finite_flat.numel() > self.factor_stats_sample_size
        ):
            sample_idx = torch.arange(
                self.factor_stats_sample_size,
                device=finite_flat.device,
                dtype=torch.long,
            )
            if self.factor_stats_sample_size > 1:
                sample_idx = (
                    sample_idx
                    * (finite_flat.numel() - 1)
                    // (self.factor_stats_sample_size - 1)
                )
            quantile_values = finite_flat[sample_idx]
            quantile_source = (
                f"sample={self.factor_stats_sample_size}/{finite_flat.numel()}"
            )

        if len(self.factor_stats_quantiles) == 0:
            quantiles = {}
        else:
            finite_values = quantile_values.detach().cpu().contiguous()
            quantiles = self._exact_quantiles(
                finite_values,
                self.factor_stats_quantiles,
            )

        return {
            "numel": total,
            "finite_numel": finite_flat.numel(),
            "min": min_value,
            "max": max_value,
            "quantile_source": quantile_source,
            "quantiles": quantiles,
        }

    def _save_factor_stats(self):
        self.factor_stats_json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.factor_stats_json_path.open("w", encoding="utf-8") as handle:
            json.dump(self._factor_stats_records, handle, indent=2, sort_keys=True)

    def _maybe_record_factor_stats(
        self,
        cache: Dict,
        eig_a: torch.Tensor,
        eig_b: torch.Tensor,
        joint_eigs: torch.Tensor,
        soft_lambda: float,
    ):
        if not self.debug_factor_stats:
            return

        cache_key = id(cache)
        if cache_key in self._factor_stats_recorded:
            return
        self._factor_stats_recorded.add(cache_key)

        layer_name = cache.get("layer_name", "<unknown>")
        self._factor_stats_records["layers"].append({
            "layer_name": str(layer_name),
            "lambda": float(soft_lambda),
            "a": self._distribution_stats(eig_a),
            "b": self._distribution_stats(eig_b),
            "ba": self._distribution_stats(joint_eigs),
        })
        self._save_factor_stats()

    def _save_grad_norm_stats(self):
        self.grad_norm_stats_json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.grad_norm_stats_json_path.open("w", encoding="utf-8") as handle:
            json.dump(self._grad_norm_records, handle, indent=2, sort_keys=True)

    def _record_grad_norm(
        self,
        cache: Optional[Dict],
        grad_before: torch.Tensor,
        grad_after: Optional[torch.Tensor],
    ):
        if not self.debug_grad_norm_stats:
            return

        before_flat = grad_before.detach().float().reshape(-1)
        before_norm = before_flat.norm().item()
        if grad_after is None:
            after_norm = None
            norm_ratio = None
            cosine = None
        else:
            after_flat = grad_after.detach().float().reshape(-1)
            after_norm = after_flat.norm().item()
            norm_ratio = (after_norm / before_norm) if before_norm > 0 else None
            denom = before_norm * after_norm
            cosine = (
                torch.dot(before_flat, after_flat).item() / denom
                if denom > 0 else None
            )

        layer_name = cache.get("layer_name", "<unknown>") if cache else "<unknown>"
        self._grad_norm_records["steps"].append({
            "step": self._grad_norm_step,
            "layer_name": str(layer_name),
            "grad_norm_before": before_norm,
            "grad_norm_after": after_norm,
            "norm_ratio": norm_ratio,
            "cosine_sim": cosine,
        })
        self._save_grad_norm_stats()

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
        self._maybe_record_factor_stats(cache, eig_a, eig_b, joint_eigs, soft_lambda)
        denom = 1.0 + float(soft_lambda) * joint_eigs
        preconditioned = q_b @ (coeffs / denom.clamp(min=1e-12)) @ q_a.T
        return preconditioned.to(dtype=tensor.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            # Adam 保持可选 closure 约定：如果调用方提供 closure，
            # 就在启用梯度的情况下重新计算 loss。
            with torch.enable_grad():
                loss = closure()

        self._grad_norm_step += 1

        for group in self.param_groups:
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

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    # 只有矩阵权重才有 K-FAC A/B 因子。
                    # bias、标量以及没有梯度的参数会退回普通 Adam 行为。
                    continue

                grad_proj = None
                if p in cache_map:
                    # 第一分支：使用主缓存执行第 3.2 节的 K-FAC 软求解。
                    # 如果缓存不完整，辅助函数会返回 None，
                    # 此处不会修改该参数的梯度。
                    _grad_before = p.grad.detach()
                    grad_proj = self._soft_kfac_precondition(
                        p.grad,
                        cache_map[p],
                        soft_lambda=soft_lambda,
                    )
                    self._record_grad_norm(cache_map[p], _grad_before, grad_proj)

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
                    # Adam 会在 super().step() 内部读取 p.grad。
                    # 因此这里原地写回预条件化后的梯度，
                    # 让 Adam 的动量从投影/软求解后的梯度更新。
                    p.grad.copy_(grad_proj)
        return super().step(closure)
