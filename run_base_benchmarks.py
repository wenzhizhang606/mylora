import os
import random
import torch
import numpy as np
import argparse
from dotenv import load_dotenv

load_dotenv()

LOCAL_DATASETS_DIR = os.getenv("HF_DATASETS_DIR", "/data1/zwz/datasets")
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "/data1/zwz"))
os.environ.setdefault("HF_DATASETS_CACHE", LOCAL_DATASETS_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.getenv("HF_HUB_CACHE", LOCAL_DATASETS_DIR))
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datasets as hf_datasets
from datasets import DownloadConfig

_ORIGINAL_LOAD_DATASET = hf_datasets.load_dataset
_LOCAL_DATASET_PATHS = {
    "cais/mmlu": ("mmlu", "cais/mmlu"),
    "hails/mmlu_no_train": ("mmlu", "mmlu_no_train", "hails/mmlu_no_train"),
    "google/IFEval": ("ifeval", "IFEval", "google/IFEval"),
    "truthfulqa/truthful_qa": ("truthful_qa", "truthfulqa/truthful_qa"),
    "openai/gsm8k": ("gsm8k", "openai/gsm8k"),
    "allenai/ai2_arc": ("ai2_arc", "allenai/ai2_arc"),
}

def _offline_enabled():
    return os.environ.get("HF_DATASETS_OFFLINE", "").lower() in {"1", "true", "yes", "on"}


def _candidate_local_dataset_paths(path):
    local_names = _LOCAL_DATASET_PATHS.get(path)
    if local_names is None:
        local_names = ()
    elif isinstance(local_names, str):
        local_names = (local_names,)

    candidates = []

    def add(candidate):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if os.path.isabs(path):
        add(path)

    for local_name in local_names:
        add(local_name if os.path.isabs(local_name) else os.path.join(LOCAL_DATASETS_DIR, local_name))

    add(os.path.join(LOCAL_DATASETS_DIR, os.path.basename(path)))
    add(os.path.join(LOCAL_DATASETS_DIR, path.replace("/", "__")))
    add(os.path.join(LOCAL_DATASETS_DIR, path.replace("/", "_")))
    return candidates


def _redirect_to_local_dataset(path):
    if path not in _LOCAL_DATASET_PATHS:
        print(f"[datasets offline] No local redirect configured for: {path}")
        return path

    candidates = _candidate_local_dataset_paths(path)
    for local_path in candidates:
        if os.path.exists(local_path):
            print(f"[datasets offline] {path} -> {local_path}")
            return local_path

    message = (
        f"[datasets offline] Local dataset not found for {path}.\n"
        f"  HF_DATASETS_DIR={LOCAL_DATASETS_DIR}\n"
        f"  Checked:\n    " + "\n    ".join(candidates)
    )
    if _offline_enabled():
        raise FileNotFoundError(
            message
            + "\nOffline mode is enabled, so the script will not download it from Hugging Face Hub."
        )

    print(message)
    return path


def load_dataset_local_first(*args, **kwargs):
    if args:
        args = (_redirect_to_local_dataset(args[0]),) + args[1:]
    elif "path" in kwargs:
        kwargs["path"] = _redirect_to_local_dataset(kwargs["path"])

    if _offline_enabled():
        kwargs.setdefault("download_config", DownloadConfig(local_files_only=True))
    return _ORIGINAL_LOAD_DATASET(*args, **kwargs)


hf_datasets.load_dataset = load_dataset_local_first

# lm_eval 相关引入
from lm_eval import simple_evaluate
from lm_eval.models.vllm_causallms import VLLM 

# 工具与自定义模块引入
from utils import print_time, save_clean_results
from easyeditor.util import HyperParams
from easyeditor.mymodels.tools.tracker import ExperimentTracker

API_KEY = os.getenv("API_KEY")

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


if __name__ =="__main__":
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
        gpu_memory_utilization=0.7,                    # 限制显存占用比例，留出部分显存防止 OOM
        trust_remote_code=True
    )

    print_time("Begin Capability Eval Time")
    HF_DATASETS_DIR = os.getenv("HF_DATASETS_DIR")
    # ── Task definitions ──────────────────────────────────────────
    tasks_with_config = {
        "mmlu":           {"shots": 5, "batch_size": "auto"},
        "ifeval":       {"shots": 0, "batch_size": "auto"},
        "truthfulqa_mc2": {"shots": 0, "batch_size": "auto"},
        "gsm8k_cot":    {"shots": 8, "batch_size": "auto"}, 
        "arc_challenge": {"shots": 25, "batch_size": "auto"},
    }

    results = {"results": {}}

    for task_name, config in tasks_with_config.items():
        print(f"Running {task_name} (Shots: {config['shots']}, Batch: {config['batch_size']})...")

        _results = simple_evaluate(
            model=lm_wrapper,
            tasks=[task_name],
            limit=args.eval_num,
            num_fewshot=config['shots'],
            batch_size=config['batch_size'],
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.apply_chat_template,
            #confirm_run_unsafe_code=True,  
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
