


def xx():
    print("挂载lora")
    model.config.use_cache = False
    model.enable_input_require_grads()

    if hparams.lora_type == "lora":
        ConfigClass = LoraConfig
    elif hparams.lora_type == "adalora":
        ConfigClass = AdaLoraConfig
    else:
        raise ValueError(f"不支持的 lora_type: {hparams.lora_type}")

    peft_config = ConfigClass(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.lora_rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
        target_modules=hparams.target_modules,
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()


def apply_limit_grad_lora_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CrispLoRAHyperParams,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs,
) -> AutoModelForCausalLM:
    tracker = kwargs.get("tracker", None) 

    if tok.padding_side != "right":
        tok.padding_side = "right"

    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]

    peft_model, optimizer_lora, optimizer_leak, _ = wrap_model_and_build_projected_optimizer(
        model, tok, hparams
    )

    device = next(peft_model.parameters()).device

    texts = [
        r["prompt"].format(r.get("subject", "")) if "{}" in r["prompt"] else r["prompt"]
        for r in requests
    ]
    targets = [r["target_new"] for r in requests]

    peft_model.train()
    for step in range(hparams.num_steps):
        total_loss = 0.0
        for txt_batch, tgt_batch in zip(
            _chunks(texts, hparams.batch_size),
            _chunks(targets, hparams.batch_size),
        ):
            optimizer_lora.zero_grad()
            optimizer_leak.zero_grad()

            loss = _compute_loss(peft_model, tok, txt_batch, tgt_batch, device, hparams)

            if loss.item() >= 1e-3:
                loss.backward()
                optimizer_lora.step()
                optimizer_leak.step()

            total_loss += loss.item()

        num_batches = max(1, math.ceil(len(texts) / hparams.batch_size))
        avg_loss = total_loss / num_batches
        if tracker is not None:
            tracker.log({"LOSS": avg_loss})
        print(f"[GradLoRA] Step {step+1}/{hparams.num_steps}  loss={avg_loss:.4f}")
        if avg_loss < 1e-3:
            print("[GradLoRA] 损失收敛，提前结束训练")
            break

    peft_model = peft_model.merge_and_unload()
    return peft_model