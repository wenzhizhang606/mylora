#!/usr/bin/env python3
"""Collect mylora benchmark logs into one JSON report.

This script is intentionally small and dependency-free: it only uses the Python
standard library, so it can run in the same environment as the rest of this
repository without requiring pandas/PyYAML.

The previous version of this helper assumed a different project layout:
``saves/unlearn/<run>/evals/*_SUMMARY.json``. This repository writes benchmark
artifacts under ``logs/<run_name>/``:

- ``capability.json`` from ``run_base_benchmarks.py`` and
  ``run_base_benchmarks_vllm.py``.
- ``mean_metrics.json`` and ``results.json`` from ``run_edited_benchmarks.py``.
- ``results_pending_judge.json`` while edited evaluation is still in progress.

The collector treats each directory under ``logs/`` as one run, flattens useful
metrics, and optionally groups or ranks runs.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any


# Metrics in this project are mostly accuracies/scores, so larger is usually
# better. Keep an explicit allowlist for common names and fall back to "higher".
HIGHER_IS_BETTER = {
    "acc",
    "acc_norm",
    "exact_match",
    "flexible-extract",
    "locality.neighborhood_acc",
    "mean_acc",
    "post.locality.neighborhood_acc",
    "post.rephrase_acc",
    "post.rewrite_acc",
    "rephrase_acc",
    "rewrite_acc",
    "safety_rewrite_safety_acc",
}


# Perplexity and loss-like metrics should be minimized.
LOWER_IS_BETTER = {
    "loss",
    "perplexity",
    "post.rewrite_ppl",
    "pre.rewrite_ppl",
    "rewrite_ppl",
}


# These group fields are derived from the log directory name. The parser is
# heuristic because README examples use names such as:
#   llama3-8b_v2_grad_32_zsre_0_3_10000_0_0005
#   llama3-8b_crispedit_zsre_0_5_10000
VALID_GROUP_FIELDS = (
    "run_name",
    "model",
    "method",
    "data_type",
    "energy_threshold",
    "cache_sample_num",
    "lr",
    "eval_kind",
)


ARTIFACT_FILES = (
    "capability.json",
    "mean_metrics.json",
    "results.json",
    "results_pending_judge.json",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options.

    Keep arguments close to the analyze-results skill text: users can select
    exact runs, glob patterns, filters, grouping fields, and a best metric.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Collect mylora logs/<run_name>/ artifacts, flatten metrics, and "
            "optionally group/rank selected runs."
        )
    )
    parser.add_argument(
        "--results-root",
        default="logs",
        help="Repo-relative or absolute directory containing one log directory per run.",
    )
    parser.add_argument(
        "--doc",
        help="Optional experiment/notes document path for metadata only.",
    )
    parser.add_argument(
        "--script",
        help="Optional script path for metadata only.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Exact log directory name to include. Repeat as needed.",
    )
    parser.add_argument(
        "--run-pattern",
        action="append",
        default=[],
        help="Glob pattern matched against log directory names. Repeat as needed.",
    )
    parser.add_argument("--model", action="append", default=[], help="Filter parsed model.")
    parser.add_argument("--method", action="append", default=[], help="Filter parsed method.")
    parser.add_argument(
        "--data-type",
        action="append",
        default=[],
        help="Filter parsed data_type, for example zsre, counterfact, or wiki.",
    )
    parser.add_argument(
        "--eval-kind",
        action="append",
        default=[],
        choices=("base", "edited", "in_progress", "unknown"),
        help="Filter by detected artifact type.",
    )
    parser.add_argument(
        "--group-by",
        action="append",
        default=[],
        help=(
            "Comma-separated grouping fields. Valid fields: "
            + ", ".join(VALID_GROUP_FIELDS)
        ),
    )
    parser.add_argument(
        "--best-by",
        help=(
            "Flattened metric name used to choose a best run, e.g. "
            "mmlu.acc, post.rewrite_acc, or post.locality.neighborhood_acc."
        ),
    )
    parser.add_argument(
        "--best-direction",
        choices=("auto", "higher", "lower"),
        default="auto",
        help="Ranking direction for --best-by.",
    )
    parser.add_argument(
        "--format",
        choices=("json",),
        default="json",
        help="Output format. JSON is intended for agent consumption.",
    )
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    """Find the repository root by walking upward until `.git` is found."""

    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return start.resolve()


def normalize_optional_path(repo_root: Path, raw_path: str | None) -> str | None:
    """Resolve optional metadata paths relative to the repository root."""

    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


def expand_group_fields(values: list[str]) -> list[str]:
    """Normalize repeated/comma-separated --group-by values."""

    fields: list[str] = []
    for value in values:
        for field in value.split(","):
            field = field.strip()
            if not field:
                continue
            if field not in VALID_GROUP_FIELDS:
                raise ValueError(
                    f"Unsupported group field '{field}'. Valid fields: "
                    + ", ".join(VALID_GROUP_FIELDS)
                )
            if field not in fields:
                fields.append(field)
    return fields


def parse_run_name(run_name: str) -> dict[str, Any]:
    """Extract useful fields from this repo's log directory names.

    This is intentionally heuristic. README examples are generated from
    `run_crispedit.py` and benchmark scripts, not from a strict schema.
    Unknown pieces are left as None instead of guessed aggressively.
    """

    parts = [part for part in run_name.split("_") if part]
    parsed: dict[str, Any] = {
        "run_name": run_name,
        "model": None,
        "method": None,
        "data_type": None,
        "energy_threshold": None,
        "cache_sample_num": None,
        "lr": None,
    }
    if not parts:
        parsed["parse_warning"] = "Empty run name."
        return parsed

    parsed["model"] = parts[0]

    # Common data type tokens used in README/utils.py.
    data_types = {
        "zsre",
        "zsre10k",
        "zsre163k",
        "counterfact",
        "multi_counterfact",
        "wiki",
        "safeedit_train",
        "safeedit_test",
    }

    data_index = next((i for i, token in enumerate(parts) if token in data_types), None)
    if data_index is not None:
        parsed["data_type"] = parts[data_index]
        method_tokens = parts[1:data_index]
        parsed["method"] = "_".join(method_tokens) if method_tokens else None
        tail = parts[data_index + 1 :]

        # run_crispedit.py encodes decimals by replacing "." with "_", so an
        # energy threshold of 0.5 may appear as "0_5".
        if len(tail) >= 2 and tail[0].isdigit() and tail[1].isdigit():
            parsed["energy_threshold"] = f"{tail[0]}.{tail[1]}"
            tail = tail[2:]
        elif tail:
            parsed["energy_threshold"] = tail[0]
            tail = tail[1:]

        if tail:
            parsed["cache_sample_num"] = tail[0]
        if len(tail) >= 4 and tail[-4].isdigit() and tail[-3].isdigit():
            # Example: 0_0005 -> 0.0005
            parsed["lr"] = f"{tail[-4]}.{tail[-3]}{tail[-2]}{tail[-1]}"
        elif len(tail) >= 2:
            parsed["lr"] = "_".join(tail[1:])
    else:
        parsed["method"] = "_".join(parts[1:]) if len(parts) > 1 else None
        parsed["parse_warning"] = "Could not find a known data_type token."

    return parsed


def load_json(path: Path) -> tuple[Any | None, str | None]:
    """Load a JSON file and return either (data, None) or (None, error)."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except FileNotFoundError:
        return None, "File not found."
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"
    except OSError as exc:
        return None, f"OS error: {exc}"


def flatten_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested JSON metrics into dot-separated numeric keys.

    Only scalar int/float values are returned. Lists and strings are ignored
    because averaging/choosing best from them would need task-specific rules.
    """

    metrics: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(flatten_metrics(child, next_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        metrics[prefix] = float(value)
    return metrics


def detect_artifacts(run_dir: Path) -> dict[str, str | None]:
    """Return existing artifact paths for one logs/<run_name> directory."""

    artifacts: dict[str, str | None] = {}
    for filename in ARTIFACT_FILES:
        path = run_dir / filename
        artifacts[filename] = str(path.resolve()) if path.exists() else None
    return artifacts


def detect_eval_kind(artifacts: dict[str, str | None]) -> str:
    """Classify a run by the files produced by this repository's evaluators."""

    if artifacts.get("capability.json"):
        return "base"
    if artifacts.get("mean_metrics.json"):
        return "edited"
    if artifacts.get("results_pending_judge.json"):
        return "in_progress"
    return "unknown"


def build_existing_row(run_name: str, run_dir: Path) -> dict[str, Any]:
    """Build one output row from an existing logs/<run_name> directory."""

    parsed = parse_run_name(run_name)
    artifacts = detect_artifacts(run_dir)
    eval_kind = detect_eval_kind(artifacts)
    metrics: dict[str, float] = {}
    errors: list[str] = []
    raw_artifact_used: str | None = None

    # Prefer compact summary artifacts. The collector does not open raw
    # results.json unless there is no summary, because it can be very large.
    for filename in ("capability.json", "mean_metrics.json"):
        artifact_path = artifacts.get(filename)
        if not artifact_path:
            continue
        data, error = load_json(Path(artifact_path))
        if error:
            errors.append(f"{filename}: {error}")
            continue
        raw_artifact_used = artifact_path
        metrics.update(flatten_metrics(data))
        break

    if eval_kind == "in_progress":
        errors.append("Run has results_pending_judge.json but no final mean_metrics.json yet.")
    elif eval_kind == "unknown":
        errors.append("No known result artifact found in this log directory.")

    if parsed.get("parse_warning"):
        errors.append(parsed["parse_warning"])

    return {
        "run_name": run_name,
        "exists": True,
        "requested_exact": False,
        "run_dir": str(run_dir.resolve()),
        "eval_kind": eval_kind,
        "parsed": parsed,
        "artifacts": artifacts,
        "metrics": metrics,
        "metric_source": raw_artifact_used,
        "errors": errors,
    }


def build_missing_row(run_name: str) -> dict[str, Any]:
    """Represent an explicitly requested run that does not exist locally."""

    return {
        "run_name": run_name,
        "exists": False,
        "requested_exact": True,
        "run_dir": None,
        "eval_kind": "missing",
        "parsed": parse_run_name(run_name),
        "artifacts": {},
        "metrics": {},
        "metric_source": None,
        "errors": ["Requested exact run was not found under results_root."],
    }


def matches_patterns(run_name: str, patterns: list[str]) -> bool:
    """Match a run name against zero or more glob patterns."""

    if not patterns:
        return True
    return any(fnmatch.fnmatch(run_name, pattern) for pattern in patterns)


def matches_filters(row: dict[str, Any], args: argparse.Namespace) -> bool:
    """Apply parsed-field filters to one row."""

    parsed = row["parsed"]
    field_map = {
        "model": args.model,
        "method": args.method,
        "data_type": args.data_type,
        "eval_kind": args.eval_kind,
    }
    for field, expected_values in field_map.items():
        actual = row["eval_kind"] if field == "eval_kind" else parsed.get(field)
        if expected_values and actual not in expected_values:
            return False
    return True


def infer_direction(metric_name: str, configured: str) -> str:
    """Choose whether larger or smaller metric values are better."""

    if configured != "auto":
        return configured
    if metric_name in LOWER_IS_BETTER or metric_name.endswith("_ppl"):
        return "lower"
    if metric_name in HIGHER_IS_BETTER or metric_name.endswith("_acc"):
        return "higher"
    return "higher"


def best_row_for_metric(
    rows: list[dict[str, Any]], metric_name: str, direction: str
) -> dict[str, Any] | None:
    """Return the best row that has the requested numeric metric."""

    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        value = row["metrics"].get(metric_name)
        if isinstance(value, (int, float)):
            candidates.append((float(value), row["run_name"], row))
    if not candidates:
        return None

    if direction == "higher":
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def summarize_groups(
    rows: list[dict[str, Any]],
    group_fields: list[str],
    best_by: str | None,
    best_direction: str,
) -> list[dict[str, Any]]:
    """Create group-level coverage summaries and optional best-run choices."""

    if not group_fields:
        return []

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(
            row["run_name"]
            if field == "run_name"
            else row["eval_kind"]
            if field == "eval_kind"
            else row["parsed"].get(field)
            for field in group_fields
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple("" if part is None else str(part) for part in value)):
        group_rows = grouped[key]
        present_rows = [row for row in group_rows if row["exists"]]
        with_metrics = [row for row in present_rows if row["metrics"]]
        summary: dict[str, Any] = {
            "group": {field: value for field, value in zip(group_fields, key)},
            "count": len(group_rows),
            "present_count": len(present_rows),
            "with_metrics_count": len(with_metrics),
            "run_names": [row["run_name"] for row in group_rows],
        }
        if best_by:
            direction = infer_direction(best_by, best_direction)
            best_row = best_row_for_metric(with_metrics, best_by, direction)
            summary["best_by"] = {
                "metric": best_by,
                "direction": direction,
                "run_name": best_row["run_name"] if best_row else None,
                "metric_value": best_row["metrics"].get(best_by) if best_row else None,
            }
        summaries.append(summary)
    return summaries


def select_rows(
    discovered: dict[str, dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Select rows by exact names, patterns, or filters."""

    selected_names: list[str] = []
    seen_names: set[str] = set()
    has_filter_selectors = any(
        (args.run_pattern, args.model, args.method, args.data_type, args.eval_kind)
    )

    for run_name in args.run:
        if run_name not in seen_names:
            selected_names.append(run_name)
            seen_names.add(run_name)

    # With no selectors, inspect every local log directory. This is useful for
    # a quick overview but users should prefer narrow patterns for large logs.
    if not args.run and not has_filter_selectors:
        selected_names.extend(discovered.keys())
    elif has_filter_selectors:
        for run_name, row in discovered.items():
            if run_name in seen_names:
                continue
            if not matches_patterns(run_name, args.run_pattern):
                continue
            if not matches_filters(row, args):
                continue
            selected_names.append(run_name)
            seen_names.add(run_name)

    rows: list[dict[str, Any]] = []
    for run_name in selected_names:
        row = discovered.get(run_name)
        if row is None:
            rows.append(build_missing_row(run_name))
            continue
        row = dict(row)
        row["requested_exact"] = run_name in args.run
        rows.append(row)
    return rows


def main() -> int:
    """Program entrypoint."""

    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)

    try:
        group_fields = expand_group_fields(args.group_by)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = repo_root / results_root
    results_root = results_root.resolve()

    metadata = {
        "repo_root": str(repo_root),
        "collector_path": str(script_path),
        "results_root": str(results_root),
        "doc_path": normalize_optional_path(repo_root, args.doc),
        "script_path": normalize_optional_path(repo_root, args.script),
    }
    selection = {
        "runs": args.run,
        "run_patterns": args.run_pattern,
        "filters": {
            "model": args.model,
            "method": args.method,
            "data_type": args.data_type,
            "eval_kind": args.eval_kind,
        },
        "group_by": group_fields,
        "best_by": args.best_by,
        "best_direction": infer_direction(args.best_by, args.best_direction)
        if args.best_by
        else None,
    }

    if not results_root.exists():
        print(
            json.dumps(
                {
                    "metadata": metadata,
                    "selection": selection,
                    "summary": {"error": "results_root does not exist"},
                    "rows": [],
                    "groups": [],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    discovered: dict[str, dict[str, Any]] = {}
    for child in sorted(results_root.iterdir(), key=lambda path: path.name):
        if child.is_dir():
            discovered[child.name] = build_existing_row(child.name, child)

    rows = select_rows(discovered, args)
    present_rows = [row for row in rows if row["exists"]]
    with_metrics_rows = [row for row in present_rows if row["metrics"]]
    available_metrics = sorted(
        metric_name
        for metric_name in {
            name for row in with_metrics_rows for name in row["metrics"].keys()
        }
    )
    groups = summarize_groups(rows, group_fields, args.best_by, args.best_direction)

    output = {
        "metadata": metadata,
        "selection": selection,
        "summary": {
            "discovered_count": len(discovered),
            "selected_count": len(rows),
            "present_count": len(present_rows),
            "with_metrics_count": len(with_metrics_rows),
            "missing_count": len(rows) - len(present_rows),
            "missing_runs": [row["run_name"] for row in rows if not row["exists"]],
            "runs_without_metrics": [
                row["run_name"] for row in present_rows if not row["metrics"]
            ],
            "available_metrics": available_metrics,
        },
        "rows": rows,
        "groups": groups,
    }

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
