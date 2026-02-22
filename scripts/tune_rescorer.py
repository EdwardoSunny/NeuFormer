"""
Hyperparameter Tuning for the WFST + LLM Rescoring Pipeline
=============================================================
Performs grid search over rescoring hyperparameters (acoustic_scale,
blank_penalty, alpha) to minimise WER on a held-out dev set.

Strategy
--------
1. Pre-compute all Conformer logits once (expensive).
2. Sweep WFST + LLM rescoring hyperparams using logits (relatively fast).
3. Report best configuration.

Usage
-----
    python scripts/tune_rescorer.py \
        --model_path logs/speech_logs/conformer_improved \
        --dataset_path ptDecoder_ctc \
        --lm_path /path/to/lm_dir \
        --do_llm \
        --llm_model facebook/opt-6.7b
"""

import argparse
import itertools
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add src and scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig
from neural_decoder.evaluation import compute_wer, remove_punctuation


@dataclass
class TuneResult:
    """Result of one hyperparameter combination."""

    acoustic_scale: float
    blank_penalty: float
    alpha: float
    beam: float
    wer: float


def main():
    parser = argparse.ArgumentParser(
        description="Tune WFST + LLM rescoring hyperparameters"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--partition", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")

    # WFST
    parser.add_argument("--lm_path", type=str, required=True)
    parser.add_argument("--nbest", type=int, default=100)

    # LLM
    parser.add_argument("--do_llm", action="store_true")
    parser.add_argument("--llm_model", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--llm_cache_dir", type=str, default=None)
    parser.add_argument("--llm_device", type=str, default="cuda")

    # Output
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--cache_path", type=str, default=None, help="Path to save/load cached logits"
    )

    args = parser.parse_args()

    # Load or compute logits
    if args.cache_path and os.path.exists(args.cache_path):
        print(f"Loading cached logits from {args.cache_path}...")
        with open(args.cache_path, "rb") as f:
            utterances = pickle.load(f)
        print(f"  Loaded {len(utterances)} utterances")
    else:
        from eval_posterior_constrained import (
            extract_logits,
            load_conformer_model,
            load_gru_model,
        )

        print("Loading dataset...")
        with open(args.dataset_path, "rb") as f:
            loaded_data = pickle.load(f)

        print("Loading model...")
        with open(os.path.join(args.model_path, "args"), "rb") as f:
            model_args = pickle.load(f)

        is_conformer = model_args.get("model_type", "gru_baseline") == "transformer_ctc"
        n_days = len(loaded_data["train"])

        if is_conformer:
            model, margs = load_conformer_model(args.model_path, n_days, args.device)
        else:
            model, margs = load_gru_model(args.model_path, n_days, args.device)

        print(f"Extracting logits from {args.partition} partition...")
        utterances = extract_logits(
            model, margs, args.partition, loaded_data, args.device
        )
        print(f"  Got {len(utterances)} utterances")

        if args.cache_path:
            os.makedirs(os.path.dirname(args.cache_path) or ".", exist_ok=True)
            with open(args.cache_path, "wb") as f:
                pickle.dump(utterances, f)
            print(f"  Cached logits saved to {args.cache_path}")

    # Grid search
    acoustic_scale_values = [0.2, 0.3, 0.35, 0.4, 0.5]
    blank_penalty_values = [5.0, 7.0, 9.0, 11.0]
    alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7] if args.do_llm else [0.5]
    beam_values = [15.0, 17.0, 19.0]

    total = (
        len(acoustic_scale_values)
        * len(blank_penalty_values)
        * len(alpha_values)
        * len(beam_values)
    )
    print(f"\nGrid search: {total} combinations")

    results: List[TuneResult] = []
    best_wer = float("inf")
    best_config = None
    n_done = 0
    t0 = time.time()

    # Create pipeline once and reuse — avoids reloading FSTs each iteration
    base_config = PipelineConfig(
        lm_path=args.lm_path,
        nbest=args.nbest,
        do_llm=args.do_llm,
        llm_model_name=args.llm_model,
        llm_cache_dir=args.llm_cache_dir,
        llm_device=args.llm_device,
    )
    pipeline = DecodePipeline(base_config)
    pipeline.setup()

    for ac_scale, bp, alpha, beam in itertools.product(
        acoustic_scale_values,
        blank_penalty_values,
        alpha_values,
        beam_values,
    ):
        pipeline.update_params(
            acoustic_scale=ac_scale,
            blank_penalty=bp,
            beam=beam,
            llm_alpha=alpha,
        )

        predictions = []
        references = []
        for utt in utterances:
            result = pipeline.decode(utt["logits"])
            predictions.append(result.sentence)
            references.append(utt["transcription"])

        wer, _, _ = compute_wer(predictions, references)

        tune_result = TuneResult(
            acoustic_scale=ac_scale,
            blank_penalty=bp,
            alpha=alpha,
            beam=beam,
            wer=wer,
        )
        results.append(tune_result)
        n_done += 1

        if wer < best_wer:
            best_wer = wer
            best_config = tune_result
            elapsed = time.time() - t0
            print(
                f"  [{n_done}/{total}] NEW BEST WER={100 * wer:.2f}%  "
                f"ac={ac_scale} bp={bp} alpha={alpha} beam={beam}  "
                f"({elapsed:.1f}s)"
            )
        elif n_done % 20 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{n_done}/{total}] best so far: {100 * best_wer:.2f}%  ({elapsed:.1f}s)"
            )

    elapsed = time.time() - t0
    print(f"\nGrid search complete in {elapsed:.1f}s")
    print(f"Best WER: {100 * best_wer:.2f}%")
    print(f"Best config: {best_config}")

    # Sort by WER
    results.sort(key=lambda r: r.wer)

    # Print top-10
    print(
        f"\n{'Rank':<6} {'WER':>8} {'ac_scale':>10} {'bp':>6} {'alpha':>7} {'beam':>6}"
    )
    print("-" * 50)
    for i, r in enumerate(results[:10]):
        print(
            f"{i + 1:<6} {100 * r.wer:>7.2f}% {r.acoustic_scale:>10.2f} "
            f"{r.blank_penalty:>6.1f} {r.alpha:>7.2f} {r.beam:>6.1f}"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_data = {
            "best": asdict(results[0]) if results else None,
            "top_20": [asdict(r) for r in results[:20]],
        }
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
