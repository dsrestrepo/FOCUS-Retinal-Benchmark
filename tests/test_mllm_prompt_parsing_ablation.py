from __future__ import annotations

import numpy as np
import pandas as pd

from retina_bench.ablations.mllm_prompt_parsing import (
    EXACT_ANSWER_FRAGMENT,
    _move_answer_instruction_first,
    evaluate_protocol,
    hard_prediction,
    parse_text,
)


def test_instruction_first_preserves_prompt_content() -> None:
    prompt = (
        "Clinical definition line.\n\n"
        "Question line.\n\n"
        'Respond only with "yes" or "no" (without additional commentary).'
    )
    varied = _move_answer_instruction_first(prompt)

    assert varied.startswith(EXACT_ANSWER_FRAGMENT)
    assert "Clinical definition line." in varied
    assert "Question line." in varied
    assert varied.count(EXACT_ANSWER_FRAGMENT) == 1


def test_strict_and_first_label_parsers_differ_only_on_extra_text() -> None:
    assert parse_text("yes", "strict_label") == 1.0
    assert parse_text("No.", "strict_label") == 0.0
    assert np.isnan(parse_text("Answer: yes", "strict_label"))
    assert parse_text("Answer: yes", "first_label") == 1.0


def test_binary_renormalization_and_coverage() -> None:
    frame = pd.DataFrame(
        {
            "ground_truth": [1, 0, 1],
            "prediction_text": ["yes", "Answer: no", "unclear"],
            "prob_yes": [0.2, 0.1, 0.0],
            "prob_no": [0.3, 0.4, 0.0],
        }
    )
    metrics, parsed = evaluate_protocol(
        frame, text_parser="first_label", score_strategy="binary_renorm"
    )

    assert np.allclose(parsed.loc[:1, "prob"], [0.4, 0.2])
    assert np.isnan(parsed.loc[2, "prob"])
    assert metrics["n_text_parsed"] == 2
    assert metrics["n_scored"] == 2
    assert np.isclose(metrics["parse_error_rate"], 1 / 3)
    assert np.isclose(metrics["overall_accuracy"], 2 / 3)


def test_first_token_argmax_prediction_and_missing_tokens() -> None:
    frame = pd.DataFrame(
        {
            "prediction_text": ["unparseable", "Answer: yes", "no"],
            "prob_yes": [0.3, 0.1, 0.0],
            "prob_no": [0.2, 0.4, 0.0],
        }
    )
    prediction = hard_prediction(frame, "first_token_argmax")

    assert prediction.iloc[0] == 1.0
    assert prediction.iloc[1] == 0.0
    assert np.isnan(prediction.iloc[2])
