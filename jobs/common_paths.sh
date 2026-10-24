#!/bin/bash
# Source after config/paths.env and optional credentials, from the repository root.
# Resolve paths only: do not create directories, move files, or change model settings.

# Always derive the Hub cache after loading HF_HOME so download and offline jobs agree.
export HF_HUB_CACHE="$HF_HOME/hub"

# Preserve the existing default layout under results/. A custom OUTPUT_DIR moves
# the sibling evaluation and analysis defaults to the same parent directory.
RESULTS_ROOT="$(dirname "$OUTPUT_DIR")"
export RESULTS_ROOT
export SFT_EVAL_DIR="${SFT_EVAL_DIR:-$RESULTS_ROOT/evals_sft}"
export GRPO_EVAL_DIR="${GRPO_EVAL_DIR:-$RESULTS_ROOT/evals_grpo}"
export SFT_ALL_EVAL_DIR="${SFT_ALL_EVAL_DIR:-$RESULTS_ROOT/evals_sft_all}"
export GRPO_ALL_EVAL_DIR="${GRPO_ALL_EVAL_DIR:-$RESULTS_ROOT/evals_grpo_all}"
export ANALYSIS_DIR="${ANALYSIS_DIR:-$RESULTS_ROOT/analysis}"
export FT_ANALYSIS_DIR="${FT_ANALYSIS_DIR:-$RESULTS_ROOT/analysis_ft}"
export SFT_ANALYSIS_DIR="${SFT_ANALYSIS_DIR:-$RESULTS_ROOT/analysis_sft}"
export GRPO_ANALYSIS_DIR="${GRPO_ANALYSIS_DIR:-$RESULTS_ROOT/analysis_grpo}"
export SFT_ALL_ANALYSIS_DIR="${SFT_ALL_ANALYSIS_DIR:-$RESULTS_ROOT/analysis_sft_all}"
export GRPO_ALL_ANALYSIS_DIR="${GRPO_ALL_ANALYSIS_DIR:-$RESULTS_ROOT/analysis_grpo_all}"
