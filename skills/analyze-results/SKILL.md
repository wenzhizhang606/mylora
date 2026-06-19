---
name: analyze-results
description: Use when completed experiment artifacts, benchmark outputs, run metrics, or docs/experiments/*.md need to be inspected, summarized, compared, or written back into an experiment document. Trigger for requests about best runs, collect_runs.py, model_utility, TOFU metrics, RESULTS_ANALYSIS.md, missing artifacts, or updating Results, Analysis, or status sections.
---

# Analyze Experiment Results

## Workflow

1. Identify the experiment from the user prompt, experiment docs, script names, or task-name patterns.
2. Read the target experiment doc first; it owns planned runs, selection rules, script paths, and output paths.
3. Read `docs/experiments/AGENTS.md` before editing experiment docs.
4. Read `RESULTS_ANALYSIS.md` or `TOFU_METRICS.md` only when metric interpretation is needed.
5. Use the narrowest useful run set. Prefer exact task names; otherwise filter by benchmark, model, split, trainer, PEFT, LR, or glob.
6. Batch-load summary metrics with the collector instead of opening many run files manually.
7. Inspect configs, logs, or raw eval JSON only for missing artifacts, anomalous metrics, or tie-breaking.
8. Before writing, state the selection rule used for best-run decisions.
9. Update mainly `## Results`, `## Analysis`, and status. Preserve the document narrative and unrelated sections.
10. Mark status as Done only when all planned runs have local artifacts and the analysis is written.

## Collector Usage

Prefer the experiment document and referenced script over generic conventions. If `collect_runs.py` is not at the repository root, locate it with `rg --files | rg collect_runs.py` or the platform equivalent.

```bash
python collect_runs.py \
  --doc docs/experiments/YY-MM-DD-example.md \
  --script scripts/experiments/YY-MM-DD-example.sh \
  --run-pattern 'tofu_Llama-3.1-8B-Instruct_forget10_*' \
  --group-by trainer,peft \
  --best-by model_utility
```

## Rules

- Never fabricate, estimate, or silently average missing data.
- Do not name a best run until the selection rule is explicit.
- Prefer exact local artifacts over assumptions from task names.
- Preserve existing status style, checkbox style, and section tone.
- Add concise evidence and interpretation instead of rewriting wholesale.
- If artifacts are missing, report what is missing and leave status incomplete.
