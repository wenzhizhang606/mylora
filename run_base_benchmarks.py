import os
import random
import torch
import numpy as np
import argparse
from dotenv import load_dotenv

# lm_eval 相关引入
from lm_eval import simple_evaluate
from lm_eval.models.vllm_causallm import VLLM

# 工具与自定义模块引入
from utils import print_time, save_clean_results
from easyeditor.util import HyperParams
from easyeditor.mymodels.tools.tracker import ExperimentTracker

load_dotenv()
API_KEY = os.getenv("API_KEY")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edited_model_dir', required=True, type=str, default=None,
                        help='Path to edited model for evaluation.')
    parser.add_argument('--data_type', required=True, type=str, default='zsre',
                        choices=['zsre', 'counterfact', 'wiki', 'safeedit_train', 'safeedit_test'])
    parser.add_argument('--eval_num', required=False, type=int, default=200,
                        help='Number of evaluation instances per task. Default 200.')
    parser.add_argument('--alg_name', required=True, type=str, default='ft_edit',
                        help='Name of the editing algorithm used.')
    parser.add_argument('--model_name', required=True, type=str, default='gpt2-xl',
                        help='Name of the base model used.')
    parser.add_argument('--wandb_project', type=str, default='CrispEdit_EVAL',
                        help='Tracker project name (also used for swanlab).')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='Resume ID (wandb only).')
    parser.add_argument('--plat_name', type=str, default='swanlab',
                        choices=['swanlab', 'wandb', 'none'],
                        help='Tracking platform.')
    parser.add_argument('--apply_chat_template', action='store_true',
                        help='Apply chat template (for instruct models). '
                             'Leave off for base models.')
    # 新增：允许从命令行控制 vLLM 参数
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs to use for tensor parallelism in vLLM.')
    args = parser.parse_args()
    return args


def build_hparams_from_args(args):
    hparams = HyperParams()
    hparams.alg_name = args.alg_name
    hparams.model_name = args.model_name
    return hparams


def resolve_model_path(edited_model_dir_local):
    """解析绝对或相对路径，用于传递给 vLLM"""
    PREFIX_DIR = os.getenv("HF_CACHE_DIR", "")
    if PREFIX_DIR and not os.path.isabs(edited_model_dir_local):
        return os.path.join(PREFIX_DIR, edited_model_dir_local)
    return edited_model_dir_local


def run_base():
    args = get_arguments()
    hparams = build_hparams_from_args(args)
    
    model_path = resolve_model_path(args.edited_model_dir)
    run_name = args.edited_model_dir.replace("/", "_").replace("\\", "_").strip("_")

    ExperimentTracker.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(hparams),
        tracker_type=args.plat_name,
        mode=(args.plat_name != "none"),
    )

    # 3. 初始化 vLLM Wrapper
    print(f"Loading vLLM model from: {model_path}")
    lm_wrapper = VLLM(
        pretrained=model_path,
        tensor_parallel_size=args.tensor_parallel_size, # 多卡支持
        dtype="auto",                                   # 自动推断精度 (fp16/bf16)
        gpu_memory_utilization=0.85,                    # 限制显存占用比例，留出部分显存防止 OOM
        trust_remote_code=True
    )

    print_time("Begin Capability Eval Time")

    # ── Task definitions ──────────────────────────────────────────
    tasks_with_config = {
        "mmlu":           {"shots": 5, "batch": "auto","dataset_path": os.path.join(,"mmlu")},
        # "ifeval":       {"shots": 0, "batch": "auto"},
        # "truthfulqa_mc2": {"shots": 0, "batch": "auto"},
        # "gsm8k_cot":    {"shots": 8, "batch": "auto"}, # 建议 vllm 也用 auto batch size
        # "arc_challenge": {"shots": 25, "batch": "auto"},
    }

    results = {"results": {}}

    for task_name, config in tasks_with_config.items():
        print(f"Running {task_name} (Shots: {config['shots']}, Batch: {config['batch']})...")

        _results = simple_evaluate(
            model=lm_wrapper,
            tasks=[task_name],
            limit=args.eval_num,
            num_fewshot=config['shots'],
            batch_size=config['batch'],
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.apply_chat_template,
        )

        if "results" in _results:
            results["results"].update(_results["results"])
            ExperimentTracker.log(_results["results"]) # 现在 tracker 已被正确引用
        else:
            print(f"Warning: no results found for task {task_name}")

    # ── Save raw results locally ─────────────────────────────────
    log_dir = f"./logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    save_clean_results(results, log_dir)

    # Log raw results as a JSON file path
    raw_results_path = os.path.join(log_dir, "capability.json")
    print(f"Raw results saved to: {raw_results_path}")

    print_time("End Capability Eval Time")
    ExperimentTracker.finish()
