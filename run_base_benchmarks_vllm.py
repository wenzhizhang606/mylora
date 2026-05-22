import argparse
import os
import random

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from lm_eval import simple_evaluate
from lm_eval.models.utils import Collator
from lm_eval.models.vllm_causallms import VLLM
from tqdm import tqdm
from vllm import SamplingParams

try:
    from vllm.inputs import TokensPrompt
except ImportError:
    TokensPrompt = None

from easyeditor.mymodels.tools.tracker import ExperimentTracker
from easyeditor.util import HyperParams
from utils import print_time, save_clean_results

_ORIGINAL_LOAD_DATASET = hf_datasets.load_dataset
_LOCAL_DATASET_PATHS = {
    "cais/mmlu": ("mmlu", "cais/mmlu"),
    "hails/mmlu_no_train": ("mmlu", "mmlu_no_train", "hails/mmlu_no_train"),
    "google/IFEval": ("ifeval", "IFEval", "google/IFEval"),
    "truthfulqa/truthful_qa": ("truthful_qa", "truthfulqa/truthful_qa"),
    "openai/gsm8k": ("gsm8k", "openai/gsm8k"),
    "allenai/ai2_arc": ("ai2_arc", "allenai/ai2_arc"),
}

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


class SafeVLLM(VLLM):
    hf_model = None
    hf_tokenizer = None

    def set_hf_fallback(self, model_path):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        self.hf_tokenizer.padding_side = "left"
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.hf_model.eval()

    def _model_generate(self, requests=None, generate=False, max_tokens=None, stop=None, **kwargs):
        if generate or self.data_parallel_size > 1:
            return super()._model_generate(
                requests=requests,
                generate=generate,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs,
            )

        sampling_params = SamplingParams(
            temperature=0,
            prompt_logprobs=1,
            max_tokens=1,
            detokenize=False,
        )
        use_tqdm = self.batch_size == "auto"

        if TokensPrompt is not None:
            prompts = [TokensPrompt(prompt_token_ids=request) for request in requests]
            return self.model.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
                lora_request=self.lora_request,
            )

        return self.model.generate(
            prompt_token_ids=requests,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=self.lora_request,
        )

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        max_ctx_len = self.max_length - 1
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0,
            batch_fn=None,
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                full = context_enc + continuation_enc
                inp = full[-max_ctx_len:]
                ctxlen = len(context_enc) - max(0, len(full) - max_ctx_len)
                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inputs, generate=False)
            for output, ctxlen, (cache_key, _, _), inp in zip(outputs, ctxlens, chunk, inputs):
                if output.prompt_logprobs is None:
                    answer = self._hf_loglikelihood(inp, ctxlen)
                else:
                    answer = self._parse_logprobs(tokens=inp, outputs=output, ctxlen=ctxlen)
                res.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def _hf_loglikelihood(self, tokens, ctxlen):
        if self.hf_model is None:
            raise RuntimeError("vLLM returned prompt_logprobs=None and HF fallback is not initialized.")

        device = next(self.hf_model.parameters()).device
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = self.hf_model(input_ids=input_ids).logits

        continuation_start = max(ctxlen, 1)
        continuation_tokens = input_ids[:, continuation_start:]
        continuation_logits = logits[:, continuation_start - 1 : -1, :]
        log_probs = torch.log_softmax(continuation_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, continuation_tokens.unsqueeze(-1)).squeeze(-1)
        continuation_logprob = token_log_probs.sum().item()

        if continuation_tokens.numel() == 0:
            is_greedy = False
        else:
            greedy_tokens = continuation_logits.argmax(dim=-1)
            is_greedy = bool(torch.equal(greedy_tokens, continuation_tokens))

        return continuation_logprob, is_greedy


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
        local_path = local_name if os.path.isabs(local_name) else os.path.join(LOCAL_DATASETS_DIR, local_name)
        add(local_path)

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


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edited_model_dir", required=True, type=str, default=None)
    parser.add_argument(
        "--data_type",
        required=True,
        type=str,
        default="zsre",
        choices=["zsre", "counterfact", "wiki", "safeedit_train", "safeedit_test"],
    )
    parser.add_argument("--eval_num", required=False, type=int, default=200)
    parser.add_argument("--alg_name", required=True, type=str, default="ft_edit")
    parser.add_argument("--model_name", required=True, type=str, default="gpt2-xl")
    parser.add_argument("--wandb_project", type=str, default="CrispEdit_EVAL")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument(
        "--plat_name",
        type=str,
        default="swanlab",
        choices=["swanlab", "wandb", "none"],
    )
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    parser.add_argument("--capability_batch_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    return parser.parse_args()


def build_hparams_from_args(args):
    hparams = HyperParams()
    hparams.alg_name = args.alg_name
    hparams.model_name = args.model_name
    return hparams


def resolve_model_path(edited_model_dir_local):
    prefix_dir = os.getenv("HF_CACHE_DIR", "")
    if prefix_dir and not os.path.isabs(edited_model_dir_local):
        return os.path.join(prefix_dir, edited_model_dir_local)
    return edited_model_dir_local


if __name__ == "__main__":
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

    print(f"Loading vLLM model from: {model_path}")
    lm_wrapper = SafeVLLM(
        pretrained=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.capability_batch_size,
        max_batch_size=args.capability_batch_size,
        max_num_seqs=args.capability_batch_size,
        max_model_len=args.max_model_len,
        dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    lm_wrapper.set_hf_fallback(model_path)

    print_time("Begin Capability Eval Time")

    tasks_with_config = {
        "mmlu": {"shots": 5, "batch_size": args.capability_batch_size},
    }
    results = {"results": {}}

    for task_name, config in tasks_with_config.items():
        print(f"Running {task_name} (Shots: {config['shots']}, Batch: {config['batch_size']})...")

        task_results = simple_evaluate(
            model=lm_wrapper,
            tasks=[task_name],
            limit=args.eval_num,
            num_fewshot=config["shots"],
            batch_size=config["batch_size"],
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.apply_chat_template,
        )

        if "results" in task_results:
            results["results"].update(task_results["results"])
            ExperimentTracker.log(task_results["results"])
        else:
            print(f"Warning: no results found for task {task_name}")

    log_dir = f"./logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    save_clean_results(results, log_dir)

    raw_results_path = os.path.join(log_dir, "capability.json")
    print(f"Raw results saved to: {raw_results_path}")

    print_time("End Capability Eval Time")
    ExperimentTracker.finish()
