# Ablation jobs

These jobs are additive: they do not edit or overwrite the main evaluation
scripts, `config/mllm_config.yaml`, predictions, or checkpoints.

Submit them from the repository root. Slurm executes a copied script under
`/var/spool/slurmd`, so every job changes to `$SLURM_SUBMIT_DIR` before reading
the same `config/paths.env` used by the paper jobs. If submission must happen
from another directory, export `FOCUS_REPO_ROOT=/absolute/path/to/RetLLM`.

## Prompt and parsing sensitivity

The base-prompt predictions already in `$OUTPUT_DIR` are reused. The GPU array
runs only `role` and `instruction_first`:

```bash
prompt_job=$(sbatch --parsable jobs/ablations/run_mllm_prompt_sensitivity.sh)
sbatch --dependency=afterok:"$prompt_job" jobs/ablations/run_mllm_parsing_sensitivity.sh
```

The configured grid has 6 models x 2 new prompts = 12 array elements. If that
grid changes, update the `#SBATCH --array` upper bound accordingly. Each
element loads one model once, evaluates all valid dataset/task pairs, skips
completed CSVs, and resumes a time-limited run from `.partial.csv`.

Response interpretation is post-hoc and therefore uses a CPU job. Each
protocol specifies (a) a text parser for the hard label used by accuracy and
(b) a probability score used by AUROC, AUPRC, ECE, and Brier score:

- `regex_binary_normalized`: first `yes`/`no` occurrence plus
  `p(yes)/(p(yes)+p(no))`; this is the primary rebuttal protocol;
- `regex_raw_yes`: the same regex plus full-vocabulary first-token `p(yes)`;
- `strict_regex_binary`: an exact `yes`/`no` response plus the
  binary-normalized score;
- `first_token_binary`: hard prediction from `p(yes) >= p(no)` plus the same
  binary-normalized score.

The analysis writes one compact LaTeX/CSV table with two panels: prompt
sensitivity under `regex_binary_normalized`, and response-extraction/scoring
sensitivity on the base prompt. It reports AUROC, ECE, accuracy, and the
invalid/unparseable decision rate.

It also writes `response_accuracy_coverage_table.{csv,tex}` for a focused
base-prompt comparison of first-token yes/no argmax, a permissive
first-yes/no response regex, and the strict
full-response regex. The table reports captured-only accuracy, full-set
accuracy with unmatched responses counted as errors, coverage, and the
unmatched/error rate.

The same count-based analysis is written per model to
`response_accuracy_coverage_by_model_table.{csv,tex}`.

## Zero-shot VLM label-prompt sensitivity

The main zero-shot predictions in `$OUTPUT_DIR` are reused as the base prompt.
The additive GPU job evaluates three new task-label pairs: bare class labels,
short with/without descriptions, and retinal-finding statements.

```bash
vlm_prompt_job=$(sbatch --parsable jobs/ablations/run_vlm_zero_shot_prompt_sensitivity.sh)
sbatch --dependency=afterok:"$vlm_prompt_job" jobs/ablations/analyze_vlm_zero_shot_prompt_sensitivity.sh
```

The array has one element per VLM (six total). Each element loads its model
once and, for each dataset/task pair, computes image embeddings once for all
three new prompt variants. The CPU analysis writes per-configuration metrics
and one per-model table containing AUROC, ECE, the composite
demographic fairness gap, and the composite image-quality gap.

## LoRA sensitivity

The two LoRA sensitivity YAML files use the same configuration concepts, with
singular or plural field names depending on whether the grid contains one or
several training values:

- `models`: Hugging Face model ids included in the ablation.
- `train_dataset` or `train_datasets`: training source(s). A list creates a
  separate adapter grid for each listed dataset.
- `task` or `train_tasks`: training task coverage. Supported values are
  `binary_dr`, `referable_dr`, and `glaucoma`; `task: all` trains one adapter
  jointly across every task available in the training dataset.
- `prompt_strategy`: training-prompt coverage. `base`, `role`, and `cot` use
  one formulation; `all` includes all three formulations in the same adapter.
- `quantization`: model-loading precision (`4b`, `8b`, or `16b`).
- `epochs`: number of passes over the generated training examples.
- `variants`: LoRA configurations to train. Each entry contains:
  - `name`: unique checkpoint/result identifier;
  - `panel`: analysis group (`rank`, `alpha`, or `target_modules`);
  - `value`: label displayed in the analysis table;
  - `lora_r`: LoRA rank;
  - `lora_alpha`: LoRA scaling parameter;
  - `lr`: learning rate;
  - `target_modules`: layers receiving adapters: `all_linear` for every linear
    layer, `attention` for q/k/v/o projections, or `query_value` for q/v only.
- `eval_datasets`: datasets used to evaluate each adapter.
- `tasks`: global evaluation-task list.
- `dataset_tasks`: valid task subset for each evaluation dataset.
- `batch_size`, `max_new_tokens`, and `split`: inference batch size, generation
  limit, and evaluation split.

`panel` and `value` only organize the output tables; training behavior is
controlled by `lora_r`, `lora_alpha`, `lr`, and `target_modules`. For SFT, the
supervised answer remains `yes` or `no` for every prompt strategy.

The original r16/alpha16/all-linear/lr2e-5 setting is the shared reference. For
both Gemma-3-27B and MedGemma-27B, the array changes exactly one factor:

- rank: 8, **16 reference**, 32 while alpha remains 16;
- alpha: 8, **16 reference**, 32 while rank remains 16;
- target modules: query/value projections as in the original LoRA paper,
  all attention projections (q/k/v/o), or the **all-linear benchmark-paper
  reference**, while rank and alpha remain 16.

All variants use BRSET/all tasks, the base/role/CoT prompt mixture, one epoch, and
learning rate 2e-5. The reference is reused, leaving 2 models x 6 new variants
= 12 training/evaluation array elements.

```bash
lora_train_job=$(sbatch --parsable jobs/ablations/train_lora_sensitivity.sh)
lora_eval_job=$(sbatch --parsable --dependency=afterok:"$lora_train_job" jobs/ablations/evaluate_lora_sensitivity.sh)
sbatch --dependency=afterok:"$lora_eval_job" jobs/ablations/analyze_lora_sensitivity.sh
```

Training uses a thin launcher around the same SFT entrypoint, 4-H100 format,
one epoch, quantization, batch-size logic, prompt mixture, and dataset/task
loader as the paper job. The launcher overrides only PEFT `target_modules`.
Evaluation uses one H100 per variant and loads each model/adapter once.
The final CPU job pairs each variant with the existing r16/lr2e-5
all-task/all-prompt predictions under `results/evals_sft_all`. Override that
location with `LORA_REFERENCE_EVAL_DIR` if needed.
It writes a single three-panel table reporting macro ID and OOD AUROC/ECE.
