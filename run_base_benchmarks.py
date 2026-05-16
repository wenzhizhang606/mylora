import os
import random
import torch
import numpy as np
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["HF_HOME"] = "/data1/zwz/dataset"
os.environ["HF_DATASETS_CACHE"] = "/data1/zwz/dataset/datasets"

os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "7" # 只使用第1、2张显卡
import argparse
from utils import print_time, save_clean_results
from transformers import AutoTokenizer, AutoModelForCausalLM
from easyeditor.util import HyperParams
from easyeditor.mymodels.tools.tracker import ExperimentTracker


SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


def get_model_and_tokenizer_from_dir(edited_model_dir_local):
    PREFIX_DIR = os.getenv("HF_CACHE_DIR", "")
    if PREFIX_DIR and not os.path.isabs(edited_model_dir_local):
        edited_model_dir = os.path.join(PREFIX_DIR, edited_model_dir_local)
    else:
        edited_model_dir = edited_model_dir_local
    tokenizer = AutoTokenizer.from_pretrained(edited_model_dir)
    model = AutoModelForCausalLM.from_pretrained(edited_model_dir, device_map='auto')
    return model, tokenizer


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
    args = parser.parse_args()
    return args


def build_hparams_from_args(args):
    hparams = HyperParams()
    hparams.alg_name = args.alg_name
    hparams.model_name = args.model_name
    return hparams


if __name__ == "__main__":
    args = get_arguments()
    hparams = build_hparams_from_args(args)
    model, tokenizer = get_model_and_tokenizer_from_dir(args.edited_model_dir)

    run_name = args.edited_model_dir.replace("/", "_").replace("\\", "_").strip("_")

    # ---- Tracker (swanlab / wandb / none) ----
    tracker = ExperimentTracker(
        project=args.wandb_project,
        name=run_name,
        config=vars(hparams),
        tracker_type=args.plat_name,
        mode=(args.plat_name != "none"),
    )
    tracker.init()

    # Ensure left padding for log-likelihood-based tasks
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    lm_wrapper = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
    )

    print_time("Begin Capability Eval Time")

    # ── Task definitions ──────────────────────────────────────────
    tasks_with_config = {
        "mmlu":           {"shots": 5, "batch": "auto"},
        # "ifeval":       {"shots": 0, "batch": "auto"},
        # "truthfulqa_mc2": {"shots": 0, "batch": "auto"},
        # "gsm8k_cot":    {"shots": 8, "batch": 2},
        # "arc_challenge": {"shots": 25, "batch": 1},
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
            tracker.log(_results["results"])
        else:
            print(f"Warning: no results found for task {task_name}")

    # ── Save raw results locally ─────────────────────────────────
    log_dir = f"./logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    save_clean_results(results, log_dir)

    # Log raw results as a JSON file path (swanlab can track this)
    raw_results_path = os.path.join(log_dir, "capability.json")
    print(f"Raw results saved to: {raw_results_path}")

    print_time("End Capability Eval Time")
    tracker.finish()
