#!/usr/bin/env python3
"""Thin target-module override around the unchanged paper SFT entrypoint.

All data construction, model loading, LoRA configuration defaults, collation,
training arguments, and saving remain in `retina_bench.adaptation.train_sft`.
This launcher only replaces the `target_modules` value passed to PEFT.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from peft import LoraConfig as PeftLoraConfig
from retina_bench.adaptation import train_sft as paper_train_sft 


TARGET_MODULES = {
    "all_linear": "all-linear",
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # The original LoRA paper's main Transformer experiments adapt query and
    # value projections while leaving key/output projections frozen.
    "query_value": ["q_proj", "v_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--lora_target_modules",
        choices=sorted(TARGET_MODULES),
        default="all_linear",
    )
    ablation_args, paper_args = parser.parse_known_args()
    target_modules = TARGET_MODULES[ablation_args.lora_target_modules]

    def ablation_lora_config(*args, **kwargs):
        kwargs["target_modules"] = target_modules
        print(
            "LoRA ablation override: "
            f"target_modules={ablation_args.lora_target_modules} ({target_modules})"
        )
        return PeftLoraConfig(*args, **kwargs)

    # train_sft imported LoraConfig into its module namespace. Replacing only
    # that symbol leaves the paper training implementation otherwise intact.
    paper_train_sft.LoraConfig = ablation_lora_config
    sys.argv = [sys.argv[0], *paper_args]
    paper_train_sft.main()


if __name__ == "__main__":
    main()
