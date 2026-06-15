import argparse
import gc
import inspect
import os
import random

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

_STARTUP_PARSER = argparse.ArgumentParser(add_help=False)
_STARTUP_PARSER.add_argument("--local_dataset_first", action="store_true")
_STARTUP_ARGS, _ = _STARTUP_PARSER.parse_known_args()

LOCAL_DATASET_FIRST = _STARTUP_ARGS.local_dataset_first
LOCAL_DATASETS_DIR = os.getenv("HF_DATASETS_DIR")
HF_DATASETS_OFFLINE = os.getenv("HF_DATASETS_OFFLINE", "")

import datasets as hf_datasets
from datasets import DownloadConfig

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

DEFAULT_TASKS = ("mmlu",)
KNOWN_TASKS = ("ifeval", "truthfulqa_mc2", "mmlu", "gsm8k_cot", "arc_challenge")


def _offline_enabled():
    return HF_DATASETS_OFFLINE.lower() in {"1", "true", "yes", "on"}


def _local_dataset_candidates(path):
    local_names = _LOCAL_DATASET_PATHS.get(path)
    if local_names is None:
        local_names = ()
    elif isinstance(local_names, str):
        local_names = (local_names,)

    candidates = []
    if os.path.isabs(path):
        candidates.append(path)

    if not LOCAL_DATASETS_DIR:
        raise RuntimeError(
            "[datasets local-first] HF_DATASETS_DIR is not configured. "
            "Set it in .env or disable --local_dataset_first."
        )

    for local_name in local_names:
        candidates.append(local_name if os.path.isabs(local_name) else os.path.join(LOCAL_DATASETS_DIR, local_name))

    candidates.extend(
        [
            os.path.join(LOCAL_DATASETS_DIR, os.path.basename(path)),
            os.path.join(LOCAL_DATASETS_DIR, path.replace("/", "__")),
            os.path.join(LOCAL_DATASETS_DIR, path.replace("/", "_")),
        ]
    )
    return list(dict.fromkeys(candidates))


def _redirect_to_local_dataset(path):
    if path not in _LOCAL_DATASET_PATHS:
        print(f"[datasets offline] No local redirect configured for: {path}")
        return path

    candidates = _local_dataset_candidates(path)
    for local_path in candidates:
        if os.path.exists(local_path):
            print(f"[datasets offline] {path} -> {local_path}")
            return local_path

    message = (
        f"[datasets offline] Local dataset not found for {path}.\n"
        f"  HF_DATASETS_DIR={LOCAL_DATASETS_DIR or '<unset>'}\n"
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


if LOCAL_DATASET_FIRST:
    hf_datasets.load_dataset = load_dataset_local_first
    print(f"[datasets local-first] enabled; HF_DATASETS_DIR={LOCAL_DATASETS_DIR}")

# Keep lm_eval imports after the optional dataset patch so task loading sees the
# selected dataset loading policy.
from lm_eval import simple_evaluate
from lm_eval.models.utils import Collator
from lm_eval.models.vllm_causallms import VLLM
from tqdm import tqdm
from vllm import SamplingParams

try:
    from vllm.inputs import TokensPrompt
except ImportError:
    TokensPrompt = None

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


def _filter_init_kwargs_for(cls, kwargs):
    signature = inspect.signature(cls.__init__)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs

    allowed = {name for name in signature.parameters if name != "self"}
    unsupported = sorted(set(kwargs) - allowed)
    if unsupported:
        print(
            "[vLLM init] Ignoring unsupported lm_eval VLLM args: "
            + ", ".join(unsupported)
        )
    return {name: value for name, value in kwargs.items() if name in allowed}


class SafeVLLM(VLLM):
    def __init__(
        self,
        *args,
        hf_model_path=None,
        loglikelihood_max_length=None,
        oom_retry_max_model_len=2048,
        hf_fallback_device="none",
        **kwargs,
    ):
        super().__init__(*args, **_filter_init_kwargs_for(VLLM, kwargs))
        self.loglikelihood_max_length = loglikelihood_max_length
        self.oom_retry_max_model_len = oom_retry_max_model_len
        self.hf_fallback_device = hf_fallback_device
        self.hf_model_path = hf_model_path
        self.hf_model = None

        if hf_fallback_device == "none":
            print("[HF fallback] disabled; not loading a second model copy.")
        else:
            print(f"[HF fallback] configured on {hf_fallback_device}; lazy-loading only if needed.")

    def _load_hf_fallback(self):
        if self.hf_model is not None:
            return
        if self.hf_fallback_device == "none" or self.hf_model_path is None:
            raise RuntimeError(
                "vLLM returned prompt_logprobs=None, but HF fallback is disabled. "
                "This is usually a vLLM prompt_logprobs compatibility issue. "
                "Re-run with --hf_fallback_device cpu to continue slowly, or use "
                "a vLLM version/API path that returns prompt_logprobs."
            )

        model_kwargs = {"torch_dtype": "auto", "trust_remote_code": True}
        if self.hf_fallback_device == "auto":
            model_kwargs["device_map"] = "auto"
        elif self.hf_fallback_device == "cpu":
            model_kwargs["device_map"] = {"": "cpu"}

        from transformers import AutoModelForCausalLM

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_path,
            **model_kwargs,
        )
        self.hf_model.eval()

    def _truncate_for_oom_retry(self, inputs, ctxlens):
        if not self.oom_retry_max_model_len or self.oom_retry_max_model_len <= 0:
            return None

        current_max_len = max(len(inp) for inp in inputs)
        target_len = self.oom_retry_max_model_len
        if target_len >= current_max_len:
            target_len = current_max_len // 2
        if target_len <= 0 or target_len >= current_max_len:
            return None

        retry_inputs = []
        retry_ctxlens = []
        for inp, ctxlen in zip(inputs, ctxlens):
            dropped = max(0, len(inp) - target_len)
            retry_inputs.append(inp[-target_len:])
            retry_ctxlens.append(max(0, ctxlen - dropped))
        return retry_inputs, retry_ctxlens, target_len

    def _model_generate_with_oom_retry(self, inputs, ctxlens):
        try:
            return self._model_generate(requests=inputs, generate=False), inputs, ctxlens
        except RuntimeError as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and "CUDA out of memory" not in str(error):
                raise

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(inputs) > 1:
                print("[vLLM OOM] Retrying this loglikelihood chunk one request at a time.")
                all_outputs = []
                all_inputs = []
                all_ctxlens = []
                for inp, ctxlen in zip(inputs, ctxlens):
                    outputs, used_inputs, used_ctxlens = self._model_generate_with_oom_retry([inp], [ctxlen])
                    all_outputs.extend(outputs)
                    all_inputs.extend(used_inputs)
                    all_ctxlens.extend(used_ctxlens)
                return all_outputs, all_inputs, all_ctxlens

            truncated = self._truncate_for_oom_retry(inputs, ctxlens)
            if truncated is None:
                raise

            retry_inputs, retry_ctxlens, target_len = truncated
            print(f"[vLLM OOM] Retrying with loglikelihood prompt length <= {target_len} tokens.")
            return self._model_generate_with_oom_retry(retry_inputs, retry_ctxlens)

    def _model_generate(self, requests=None, generate=False, max_tokens=None, stop=None, **kwargs):
        if generate or getattr(self, "data_parallel_size", 1) > 1:
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

        generate_kwargs = {
            "sampling_params": sampling_params,
            "use_tqdm": use_tqdm,
        }
        if self.lora_request is not None:
            generate_kwargs["lora_request"] = self.lora_request

        try:
            return self.model.generate(prompt_token_ids=requests, **generate_kwargs)
        except TypeError:
            if TokensPrompt is None:
                raise
            prompts = [TokensPrompt(prompt_token_ids=request) for request in requests]
            return self.model.generate(prompts, **generate_kwargs)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        max_ctx_len = self.max_length - 1
        if self.loglikelihood_max_length and self.loglikelihood_max_length > 0:
            max_ctx_len = min(max_ctx_len, self.loglikelihood_max_length)
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 1,
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
                ctxlen = max(0, len(context_enc) - max(0, len(full) - max_ctx_len))
                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs, inputs, ctxlens = self._model_generate_with_oom_retry(inputs, ctxlens)
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
        self._load_hf_fallback()

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


def get_arguments(argv=None):
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
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--plat_name",
        type=str,
        default="swanlab",
        choices=["swanlab", "wandb", "none"],
    )
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument(
        "--local_dataset_first",
        action="store_true",
        help="Try configured local dataset paths before lm_eval's default dataset path.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.60)
    parser.add_argument("--capability_batch_size", type=str, default="1")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help='Comma-separated lm_eval tasks, or "all". Defaults to mmlu.',
    )
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--loglikelihood_max_length", type=int, default=2048)
    parser.add_argument("--oom_retry_max_model_len", type=int, default=1024)
    parser.add_argument(
        "--hf_fallback_device",
        type=str,
        default="none",
        choices=["none", "cpu", "auto"],
    )
    args = parser.parse_args(argv)
    if args.capability_batch_size != "auto":
        args.capability_batch_size = int(args.capability_batch_size)
    return args


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


def parse_tasks(tasks_arg):
    if not tasks_arg:
        return list(DEFAULT_TASKS)
    if tasks_arg.strip().lower() == "all":
        return list(KNOWN_TASKS)

    tasks = [task.strip() for task in tasks_arg.split(",") if task.strip()]
    if not tasks:
        raise ValueError("--tasks was provided but no valid task names were found.")
    return tasks


def build_tasks_with_config(args):
    default_batch_size = args.capability_batch_size
    heavy_batch_size = 1 if default_batch_size == "auto" else default_batch_size
    cot_batch_size = 2 if default_batch_size == "auto" else default_batch_size

    task_defaults = {
        "ifeval": {"shots": 0, "batch_size": default_batch_size},
        "truthfulqa_mc2": {"shots": 0, "batch_size": default_batch_size},
        "mmlu": {"shots": 5, "batch_size": default_batch_size},
        "gsm8k_cot": {"shots": 8, "batch_size": cot_batch_size},
        "arc_challenge": {"shots": 25, "batch_size": heavy_batch_size},
    }

    tasks_with_config = {}
    for task_name in parse_tasks(getattr(args, "tasks", None)):
        if task_name not in task_defaults:
            print(
                f"[lm_eval tasks] No local default for {task_name}; "
                "using 0-shot with the default batch size."
            )
        tasks_with_config[task_name] = task_defaults.get(
            task_name,
            {"shots": 0, "batch_size": default_batch_size},
        )
    return tasks_with_config


def run(args):
    hparams = build_hparams_from_args(args)

    model_path = resolve_model_path(args.edited_model_dir)
    run_name = args.edited_model_dir.replace("/", "_").replace("\\", "_").strip("_")
    vllm_batch_limit = 1 if args.capability_batch_size == "auto" else int(args.capability_batch_size)

    ExperimentTracker.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(hparams),
        tracker_type=args.plat_name,
        mode=(args.plat_name != "none" and not args.no_wandb),
    )

    try:
        print(f"Loading vLLM model from: {model_path}")
        lm_wrapper = SafeVLLM(
            pretrained=model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.capability_batch_size,
            max_batch_size=vllm_batch_limit,
            max_num_seqs=vllm_batch_limit,
            max_model_len=args.max_model_len,
            hf_model_path=model_path,
            loglikelihood_max_length=args.loglikelihood_max_length,
            oom_retry_max_model_len=args.oom_retry_max_model_len,
            hf_fallback_device=args.hf_fallback_device,
            dtype="auto",
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )

        print_time("Begin Capability Eval Time")

        results = {"results": {}}
        tasks_with_config = build_tasks_with_config(args)

        for task_name, config in tasks_with_config.items():
            print(
                f"Running {task_name} "
                f"(Shots: {config['shots']}, Batch: {config['batch_size']})..."
            )

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
    finally:
        ExperimentTracker.finish()


def main(argv=None):
    args = get_arguments(argv)
    run(args)


if __name__ == "__main__":
    main()
