import math
import os
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


# ─────────────────────────────────────────────────────────────────────────────
# KFac 投影（针对 delta_W = B @ A 整体）
# ─────────────────────────────────────────────────────────────────────────────

def _build_ab_projection_cache(model, tok, hparams):
    """计算各层 KFac 协方差，返回 layer_name -> (mask_a, mask_b)"""
    from ...models.rome.layer_stats import layer_stats_kfac_one_pass
    from ..limit_grad_lora.utils import compute_marginal_masks

    layer_names = [hparams.rewrite_module_tmp.format(l) for l in hparams.layers]
    stats_dict = layer_stats_kfac_one_pass(
        model=model, tokenizer=tok, layer_names=layer_names,
        stats_dir=STATS_DIR, ds_name=hparams.mom2_dataset,
        to_collect=["mom2"], sample_size=hparams.mom2_n_samples,
        precision=hparams.mom2_dtype, force_recompute=False,
    )

    cache = {}
    for layer_name in layer_names:
        A, B, _ = stats_dict.pop(layer_name)
        lower = hparams.model_name.lower()
        if not ("llama" in lower or "phi" in lower):
            A, B = B, A
        A, B = A.float(), B.float()
        Sa, Ua = torch.linalg.eigh(A)
        Sb, Ub = torch.linalg.eigh(B)
        mask_a, mask_b = compute_marginal_masks(Sa, Ua, Sb, Ub, hparams.energy_threshold)
        cache[layer_name] = {"mask_a": mask_a.cpu(), "mask_b": mask_b.cpu()}
        del A, B, Sa, Sb, Ua, Ub
        torch.cuda.empty_cache()
    return cache


def _project_delta_w(delta_w: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    """
    对 delta_W (d_out, d_in) 做双侧投影，屏蔽高曲率方向。
    mask_a: (d_in, k_in)，mask_b: (d_out, k_out)
    """
    dev, dtype = delta_w.device, delta_w.dtype
    mask_a = mask_a.to(device=dev, dtype=dtype)
    mask_b = mask_b.to(device=dev, dtype=dtype)
    # 右侧（输入方向）高曲率分量
    high_right = delta_w @ (mask_a @ mask_a.T)
    # 左侧（输出方向）高曲率分量（在去掉右侧高曲率后再投影）
    delta_low = delta_w - high_right
    high_left = (mask_b @ mask_b.T) @ delta_low
    return delta_low - high_left


def _apply_ab_projection(peft_model, proj_cache: Dict, hparams):
    """
    直接将投影后的 delta_W 加到 base weight，并清零 lora_A/B。
    merge_and_unload() 后续合并时加的是零，等效于替换了 merge 过程。
    """
    for name, module in peft_model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        matched = None
        for layer_name in proj_cache:
            clean = layer_name[:-len(".weight")] if layer_name.endswith(".weight") else layer_name
            if clean in name:
                matched = layer_name
                break
        if matched is None:
            continue

        lora_A = module.lora_A["default"].weight  # (r, d_in)
        lora_B = module.lora_B["default"].weight  # (d_out, r)
        scaling = module.scaling["default"]
        base_weight = module.base_layer.weight    # (d_out, d_in)

        delta_w = (lora_B @ lora_A) * scaling
        mask_a = proj_cache[matched]["mask_a"]
        mask_b = proj_cache[matched]["mask_b"]
        delta_w_proj = _project_delta_w(delta_w, mask_a, mask_b)

        with torch.no_grad():
            base_weight.data += delta_w_proj.to(base_weight.dtype)
            lora_A.zero_()
            lora_B.zero_()

        print(f"[LoRAProj] {name}: delta_W 投影后直接写入 base weight")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams,
    **kwargs,
) -> AutoModelForCausalLM:
    """
    hparams 需要有：
        lora_rank, lora_alpha, lora_dropout, target_modules, layers,
        lr, weight_decay, num_steps, batch_size, max_length
    可选（use_projection=True 时需要）：
        use_projection, rewrite_module_tmp, mom2_dataset, mom2_n_samples,
        mom2_dtype, energy_threshold, model_name
    """
    print("正确进入apply_lora_to_model函数")
    tracker = kwargs.get("tracker", None)
    use_projection = getattr(hparams, "use_projection", False)

    if tok.padding_side != "right":
        tok.padding_side = "right"
    for i, r in enumerate(requests):
        if r["target_new"] and r["target_new"][0] != " ":
            requests[i]["target_new"] = " " + r["target_new"]

    # 投影缓存在挂 LoRA 之前基于原始权重计算
    proj_cache = None
    if use_projection:
        print("[LoRAProj] 计算 KFac 投影缓存...")
        proj_cache = _build_ab_projection_cache(model, tok, hparams)

    model.config.use_cache = False
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.lora_rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if hparams.layers else None,
        target_modules=hparams.target_modules,
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    optimizer = torch.optim.Adam(
        peft_model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay
    )

    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    device = next(peft_model.parameters()).device

    peft_model.train()
    for step in range(hparams.num_steps):
        total_loss = 0.0
        for txt_batch, tgt_batch in zip(
            _chunks(texts, hparams.batch_size),
            _chunks(targets, hparams.batch_size),
        ):
            optimizer.zero_grad()
            loss = _compute_loss(peft_model, tok, txt_batch, tgt_batch, device, hparams)
            if loss.item() >= 1e-3:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, math.ceil(len(texts) / hparams.batch_size))
        if tracker is not None:
            tracker.log({"LOSS": avg_loss})
        print(f"[LoRA] Step {step+1}/{hparams.num_steps}  loss={avg_loss:.4f}")
        if avg_loss < 1e-3:
            print("[LoRA] 损失收敛，提前结束")
            break

    # merge 前做 delta_W 投影
    if use_projection and proj_cache is not None:
        print("[LoRAProj] 对 delta_W=B@A 做 KFac 投影...")
        peft_model.eval()
        _apply_ab_projection(peft_model, proj_cache, hparams)

    return peft_model.merge_and_unload()


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _compute_loss(model, tok, texts, targets, device, hparams):
    inputs_targets = [t + tg for t, tg in zip(texts, targets)]
    encodings = tok(
        inputs_targets, return_tensors="pt",
        padding=True, truncation=True, max_length=hparams.max_length,
    ).to(device)
    labels = encodings["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    for i, prompt in enumerate(texts):
        prompt_len = len(tok(
            prompt, add_special_tokens=True,
            truncation=True, max_length=hparams.max_length
        )["input_ids"])
        labels[i, :prompt_len] = -100
    return model(**encodings, labels=labels).loss


def _chunks(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]
