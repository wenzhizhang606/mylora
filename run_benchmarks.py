"""Unified benchmark entrypoint.

Examples:
  python run_benchmarks.py --test_mode base ...
  python run_benchmarks.py --test_mode edited ...
"""

import argparse


DATA_TYPES = [
    "zsre",
    "counterfact",
    "multi_counterfact",
    "wiki",
    "safeedit_train",
    "safeedit_test",
]


def get_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Dispatch all benchmark evaluations from one entrypoint."
    )

    parser.add_argument(
        "--test_mode",
        type=str,
        default="edited",
        choices=["base", "edited"],
        help="base runs lm_eval capability benchmarks with vLLM; edited runs edit-quality benchmarks with vLLM.",
    )

    parser.add_argument(
        "--edited_model_dir",
        required=True,
        type=str,
        help="Path to the model directory for evaluation.",
    )
    parser.add_argument(
        "--data_type",
        required=True,
        type=str,
        default="zsre",
        choices=DATA_TYPES,
    )
    parser.add_argument(
        "--eval_num",
        required=False,
        type=int,
        default=None,
        help="Number of evaluation instances. Defaults depend on test mode.",
    )
    parser.add_argument(
        "--alg_name",
        required=True,
        type=str,
        default="ft_edit",
        help="Name of the editing algorithm used.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        default="gpt2-xl",
        help="Name of the base model used.",
    )

    # Edited benchmark args.
    parser.add_argument(
        "--max_length",
        required=False,
        type=int,
        default=40,
        help="Maximum length of generated sequences for edited benchmarks.",
    )
    parser.add_argument(
        "--context_type",
        required=False,
        type=str,
        default="qa_inst",
        choices=["qa_inst", "chat_temp", "no_context"],
        help="Context type for edited benchmarks.",
    )
    parser.add_argument(
        "--evaluation_criteria",
        required=False,
        type=str,
        default="exact_match",
        choices=["exact_match", "llm_judge"],
        help="Evaluation criteria for edited benchmarks.",
    )
    parser.add_argument(
        "--judge_batch_size",
        required=False,
        type=int,
        default=16,
        help="Number of generated answers to judge per batch.",
    )

    # Tracker args.
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="CrispEdit_EVAL",
        help="Tracker project name.",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="WandB run ID for resuming runs.",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable tracker logging.")
    parser.add_argument(
        "--plat_name",
        type=str,
        default=None,
        choices=["swanlab", "wandb", "none"],
        help="Tracking platform. Defaults depend on test mode.",
    )

    # Runtime args.
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply lm_eval chat template for instruct models.",
    )
    parser.add_argument(
        "--local_dataset_first",
        action="store_true",
        help="For base lm_eval tasks, try configured local dataset paths first.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="vLLM GPU memory utilization. Defaults depend on test mode.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="vLLM dtype for edited benchmarks.",
    )

    # lm_eval capability args.
    parser.add_argument(
        "--capability_batch_size",
        type=str,
        default=None,
        help='Batch size for lm_eval capability benchmarks. Use "auto" or an integer.',
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated lm_eval tasks for --test_mode base.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="vLLM max model length.",
    )
    parser.add_argument(
        "--loglikelihood_max_length",
        type=int,
        default=None,
        help="Max token length used for vLLM loglikelihood prompts.",
    )
    parser.add_argument(
        "--oom_retry_max_model_len",
        type=int,
        default=None,
        help="Retry vLLM loglikelihood OOMs with prompts truncated to this length.",
    )
    parser.add_argument(
        "--hf_fallback_device",
        type=str,
        default="none",
        choices=["none", "cpu", "auto"],
        help="HF fallback device if vLLM does not return prompt logprobs.",
    )

    return parser.parse_args(argv)


def _normalize_args(args):
    if args.test_mode == "base":
        if args.eval_num is None:
            args.eval_num = 200
        if args.plat_name is None:
            args.plat_name = "swanlab"

        if args.capability_batch_size is None:
            args.capability_batch_size = "1"
        if args.gpu_memory_utilization is None:
            args.gpu_memory_utilization = 0.60
        if args.max_model_len is None:
            args.max_model_len = 2048
        if args.loglikelihood_max_length is None:
            args.loglikelihood_max_length = args.max_model_len
        if args.oom_retry_max_model_len is None:
            args.oom_retry_max_model_len = 1024

        if args.capability_batch_size != "auto":
            args.capability_batch_size = int(args.capability_batch_size)
    else:
        if args.eval_num is None:
            args.eval_num = 3000
        if args.plat_name is None:
            args.plat_name = "wandb"
        if args.gpu_memory_utilization is None:
            args.gpu_memory_utilization = 0.90

    return args


def run(args):
    args = _normalize_args(args)

    if args.test_mode == "base":
        from run_base_benchmarks_vllm import run as run_base_vllm
        return run_base_vllm(args)
    elif args.test_mode == "edited":
        from run_edited_benchmarks import run as run_edited
        return run_edited(args)


def main(argv=None):
    args = get_arguments(argv)
    run(args)


if __name__ == "__main__":
    main()
