"""独立检查不同层 K-FAC 因子（A/B）条件数的脚本。

矩阵都是提前算好缓存到磁盘的（layer_stats_kfac_one_pass 产出），本脚本
无需加载模型/GPU，直接读取缓存文件，复现 _generalized_basis 中的阻尼逻辑，
对比阻尼前后的条件数，诊断各层 edit/cap 因子的病态程度。

缓存路径约定：
    {stats_dir}/{model_name}/{ds_name}_stats/{layer_name}_{precision}_kfac{size_suffix}.npz
（注意扩展名虽为 .npz，实际是 torch.save 的 pickle，含 {'A','B','N'}）

用法示例：
    python check_kfac_condition.py \
        --stats_dir $STATS_DIR \
        --model_name "Meta-Llama-3-8B-Instruct" \
        --base_ds wikipedia --base_sample_size 10000 \
        --task_ds zsre_mend_163k --task_sample_size 10000 \
        --layers 4,5,6,7,8 \
        --rewrite_module_tmp "model.layers.{}.mlp.down_proj.weight" \
        --precision float32 --factor_damping 1.0e-5 \
        --out kfac_condition_report.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from dotenv import load_dotenv

load_dotenv()


def kfac_cache_path(
    stats_dir: Path,
    model_name: str,
    ds_name: str,
    layer_name: str,
    precision: str,
    sample_size: Optional[int],
) -> Path:
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    return stats_dir / f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_kfac{size_suffix}.npz"


def load_kfac(path: Path):
    if not path.exists():
        return None
    loaded = torch.load(path, map_location="cpu")
    # 兼容 dict {'A','B','N'} 或直接 tuple
    if isinstance(loaded, dict):
        return loaded.get("A"), loaded.get("B"), loaded.get("N")
    return loaded  # (A, B, N)


def symmetrize(m: torch.Tensor) -> torch.Tensor:
    return 0.5 * (m + m.T)


def condition_stats(matrix: torch.Tensor, factor_damping: float) -> Dict:
    """复现 _generalized_basis 的阻尼逻辑，返回阻尼前后条件数等统计。"""
    m = symmetrize(matrix.to(torch.float32))
    n = m.shape[0]

    eigvals = torch.linalg.eigvalsh(m)
    eigvals_min = eigvals.clamp(min=1e-20).min().item()
    eigvals_max = eigvals.max().item()
    cond_raw = eigvals_max / max(eigvals_min, 1e-20)

    # 与 _generalized_basis 一致：eps = factor_damping * |diag|.mean()
    trace_scale = m.diagonal().abs().mean().clamp(min=1e-12).item()
    eps = float(factor_damping) * trace_scale
    m_reg = m + eps * torch.eye(n, dtype=m.dtype)

    eigvals_reg = torch.linalg.eigvalsh(m_reg)
    eigvals_reg_min = eigvals_reg.clamp(min=1e-20).min().item()
    eigvals_reg_max = eigvals_reg.max().item()
    cond_reg = eigvals_reg_max / max(eigvals_reg_min, 1e-20)

    return {
        "shape": list(m.shape),
        "eig_min": eigvals_min,
        "eig_max": eigvals_max,
        "cond_before": cond_raw,
        "trace_scale": trace_scale,
        "eps": eps,
        "eig_min_after": eigvals_reg_min,
        "eig_max_after": eigvals_reg_max,
        "cond_after": cond_reg,
    }


def fmt(x, mode="sci"):
    if x is None:
        return "n/a"
    if mode == "sci":
        return f"{x:.3e}"
    return f"{x:.4g}"


def report_matrix(name: str, stats: Optional[Dict]) -> str:
    if stats is None:
        return f"  {name:8s}: <无缓存，跳过>"
    return (
        f"  {name:8s}: shape={stats['shape']}  "
        f"cond(前)={fmt(stats['cond_before'])}  cond(后)={fmt(stats['cond_after'])}  "
        f"eig[min,max]=[{fmt(stats['eig_min'])}, {fmt(stats['eig_max'])}]  "
        f"eps={fmt(stats['eps'])}"
    )


def main():
    ap = argparse.ArgumentParser(description="检查各层 K-FAC 因子条件数（阻尼前后对比）")
    ap.add_argument("--stats_dir", default=os.getenv("STATS_DIR"),
                    help="K-FAC 统计根目录，默认读环境变量 STATS_DIR")
    ap.add_argument("--model_name", required=True,
                    help="模型名，如 Meta-Llama-3-8B-Instruct")
    ap.add_argument("--base_ds", default="wikipedia", help="base/cap 协方差数据集名")
    ap.add_argument("--base_sample_size", type=int, default=10000)
    ap.add_argument("--task_ds", default=None, help="task/edit 协方差数据集名（可选）")
    ap.add_argument("--task_sample_size", type=int, default=10000)
    ap.add_argument("--layers", required=True,
                    help="层号逗号分隔，如 4,5,6,7,8")
    ap.add_argument("--rewrite_module_tmp", default="model.layers.{}.mlp.down_proj.weight",
                    help="权重路径模板，与 hparams.rewrite_module_tmp 一致")
    ap.add_argument("--precision", default="float32")
    ap.add_argument("--factor_damping", type=float, default=1.0e-5,
                    help="与 ProjectedAdam.factor_damping 一致")
    ap.add_argument("--out", default=None, help="可选，将报告保存为 JSON")
    args = ap.parse_args()

    if not args.stats_dir:
        raise SystemExit("未指定 --stats_dir，且环境变量 STATS_DIR 为空。")

    stats_dir = Path(args.stats_dir)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    print(f"stats_dir       : {stats_dir}")
    print(f"model_name      : {args.model_name}")
    print(f"base (cap)      : {args.base_ds} @ {args.base_sample_size}")
    if args.task_ds:
        print(f"task (edit)     : {args.task_ds} @ {args.task_sample_size}")
    print(f"layers          : {layers}")
    print(f"precision       : {args.precision}")
    print(f"factor_damping  : {args.factor_damping}")
    print("=" * 100)

    report = {
        "config": vars(args),
        "layers": [],
    }

    for layer in layers:
        layer_name = args.rewrite_module_tmp.format(layer)
        print(f"\n[Layer {layer}] {layer_name}")

        layer_entry = {"layer": layer, "layer_name": layer_name, "factors": {}}

        # cap 因子（base 协方差）
        cap_path = kfac_cache_path(
            stats_dir, args.model_name, args.base_ds, layer_name,
            args.precision, args.base_sample_size,
        )
        cap = load_kfac(cap_path)
        if cap is not None:
            cap_A, cap_B, cap_N = cap
            for fname, mat in (("cap_A", cap_A), ("cap_B", cap_B)):
                if mat is None:
                    continue
                s = condition_stats(mat, args.factor_damping)
                layer_entry["factors"][fname] = s
                print(report_matrix(fname, s))
            if cap_N is not None:
                print(f"  cap_N    : {cap_N}")
        else:
            print(f"  <cap 缓存不存在: {cap_path}>")

        # edit 因子（task 协方差，可选）
        if args.task_ds:
            edit_path = kfac_cache_path(
                stats_dir, args.model_name, args.task_ds, layer_name,
                args.precision, args.task_sample_size,
            )
            edit = load_kfac(edit_path)
            if edit is not None:
                edit_A, edit_B, edit_N = edit
                for fname, mat in (("edit_A", edit_A), ("edit_B", edit_B)):
                    if mat is None:
                        continue
                    s = condition_stats(mat, args.factor_damping)
                    layer_entry["factors"][fname] = s
                    print(report_matrix(fname, s))
                if edit_N is not None:
                    print(f"  edit_N   : {edit_N}")
            else:
                print(f"  <edit 缓存不存在: {edit_path}>")

        report["layers"].append(layer_entry)

    print("\n" + "=" * 100)
    print("汇总（阻尼前 → 阻尼后 条件数）：")
    print(f"  {'layer':<6}{'factor':<8}{'cond_before':<16}{'cond_after':<16}{'eps':<14}")
    for le in report["layers"]:
        for fname, s in le["factors"].items():
            print(f"  {le['layer']:<6}{fname:<8}{fmt(s['cond_before']):<16}"
                  f"{fmt(s['cond_after']):<16}{fmt(s['eps']):<14}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()
