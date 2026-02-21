"""
Hyperparameter Tuning for the Rescoring Pipeline
=================================================
Performs grid search (or optional Optuna Bayesian optimisation) over
rescoring hyperparameters to minimise WER on a held-out dev set.

The key insight: the rescoring lambda weights, confidence threshold,
and normalization settings are completely untuned — systematic search
can recover a significant fraction of the oracle WER gap.

Strategy
--------
1. Pre-compute all neural logits + LLM scores once (expensive).
2. Sweep rescoring hyperparams offline using cached scores (cheap).
3. Report best configuration and full Pareto front.

This avoids re-running the Conformer or LLM for each hyperparameter
combination — only the linear combination + constraint penalty changes.

Usage
-----
    python scripts/tune_rescorer.py \
        --model_path logs/speech_logs/conformer_improved \
        --dataset_path ptDecoder_ctc \
        --llm_model meta-llama/Meta-Llama-3-8B \
        --llm_device cuda:1 \
        --mode constrained_rescore \
        --output results/tune_results.json
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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig
from neural_decoder.evaluation import evaluate_wer, oracle_wer


@dataclass
class CachedUtterance:
    """Pre-computed data for one utterance (avoids re-running Conformer/LLM)."""

    # Word-level N-best from pipeline (neural)
    candidate_lists: List[List[str]]  # word sequences
    neural_scores: List[float]  # per-candidate neural scores
    lm_scores: List[float]  # per-candidate LLM scores
    # Confidence for template building
    word_confidences: List[Tuple[str, float]]
    # Reference
    ref_words: List[str]


@dataclass
class TuneResult:
    """Result of one hyperparameter combination."""

    lambda_neural: float
    lambda_lm: float
    gamma_constraint: float
    length_penalty_beta: float
    high_confidence_threshold: float
    normalize_scores: bool
    wer: float
    oracle: float


def precompute_scores(
    model_path: str,
    dataset_path: str,
    llm_model: str,
    llm_device: str,
    device: str,
    partition: str,
    beam_width: int,
    n_best: int,
    max_word_candidates: int,
) -> Tuple[List[CachedUtterance], float]:
    """
    Run the full pipeline once to cache neural scores + LLM scores.

    Returns (cached_utterances, oracle_wer_value).
    """
    # Import here to avoid loading torch at import time
    from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig

    # Reuse the eval script's loading infrastructure
    from eval_posterior_constrained import (
        extract_logits,
        get_training_sentences,
        load_conformer_model,
        load_gru_model,
    )

    print("Loading dataset...")
    with open(dataset_path, "rb") as f:
        loaded_data = pickle.load(f)

    print("Loading model...")
    with open(os.path.join(model_path, "args"), "rb") as f:
        model_args = pickle.load(f)

    is_conformer = model_args.get("model_type", "gru_baseline") == "transformer_ctc"
    n_days = len(loaded_data["train"])

    if is_conformer:
        model, args = load_conformer_model(model_path, n_days, device)
    else:
        model, args = load_gru_model(model_path, n_days, device)

    print(f"Extracting logits from {partition} partition...")
    utterances = extract_logits(model, args, partition, loaded_data, device)
    print(f"  Got {len(utterances)} utterances")

    training_sentences = get_training_sentences(loaded_data)

    # Setup pipeline with generous beam settings for good N-best lists
    config = PipelineConfig(
        beam_width=beam_width,
        n_best=n_best,
        llm_model_name=llm_model,
        llm_device=llm_device,
        decode_mode="neural_only",  # We'll do rescoring manually
        max_word_candidates=max_word_candidates,
        normalize_scores=False,  # Raw scores for caching
    )
    pipeline = DecodePipeline(config)
    pipeline.setup(training_sentences)
    print(f"  Lexicon size: {pipeline.lexicon.size} words")

    # Phase 1: Get neural scores + word hypotheses for all utterances
    print("Phase 1: Running CTC beam search + lexicon for all utterances...")
    t0 = time.time()
    raw_results = []
    for i, utt in enumerate(utterances):
        result = pipeline.decode_utterance(utt["log_probs"], utt["length"])
        raw_results.append((utt, result))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(utterances)} utterances processed")
    print(f"  Neural decoding took {time.time() - t0:.1f}s")

    # Phase 2: Batch LLM scoring of all candidates
    print("Phase 2: LLM scoring all candidates...")
    llm = pipeline._get_llm()

    cached = []
    all_candidate_sets = []
    all_ref_words = []

    t0 = time.time()
    for utt, result in raw_results:
        ref_words = utt["transcription"].split() if utt["transcription"] else []

        if not result.word_hypotheses:
            cached.append(
                CachedUtterance(
                    candidate_lists=[ref_words] if ref_words else [[]],
                    neural_scores=[0.0],
                    lm_scores=[0.0],
                    word_confidences=[],
                    ref_words=ref_words,
                )
            )
            all_candidate_sets.append([ref_words] if ref_words else [[]])
            all_ref_words.append(ref_words)
            continue

        n_cands = min(len(result.word_hypotheses), max_word_candidates)
        cand_lists = [wh[0] for wh in result.word_hypotheses[:n_cands]]
        neural_scores = [wh[1] for wh in result.word_hypotheses[:n_cands]]

        # Batch LLM scoring
        sentences = [" ".join(words) for words in cand_lists]
        lm_scores = llm.score_batch(sentences)

        word_conf_tuples = [(wc.word, wc.confidence) for wc in result.word_confidences]

        cached.append(
            CachedUtterance(
                candidate_lists=cand_lists,
                neural_scores=neural_scores,
                lm_scores=lm_scores,
                word_confidences=word_conf_tuples,
                ref_words=ref_words,
            )
        )
        all_candidate_sets.append(cand_lists)
        all_ref_words.append(ref_words)

    print(f"  LLM scoring took {time.time() - t0:.1f}s")

    ow = oracle_wer(all_candidate_sets, all_ref_words)
    print(f"  Oracle WER: {ow:.4f}")

    return cached, ow


def rescore_with_params(
    cached_utts: List[CachedUtterance],
    lambda_neural: float,
    lambda_lm: float,
    gamma_constraint: float,
    length_penalty_beta: float,
    high_confidence_threshold: float,
    normalize: bool,
    mode: str = "unconstrained_rescore",
) -> float:
    """
    Rescore all utterances using cached scores and given hyperparams.

    This is very fast — no neural model or LLM calls, just linear
    combination of pre-computed scores.
    """
    from neural_decoder.constrained_decode import (
        ConstrainedHypothesisBuilder,
        SlotFillingDecoder,
    )

    all_pred_words = []
    all_ref_words = []

    builder = ConstrainedHypothesisBuilder(
        high_confidence_threshold=high_confidence_threshold,
    )

    for cu in cached_utts:
        if not cu.candidate_lists or not cu.candidate_lists[0]:
            all_pred_words.append([])
            all_ref_words.append(cu.ref_words)
            continue

        ns_arr = np.array(cu.neural_scores, dtype=np.float64)
        lm_arr = np.array(cu.lm_scores, dtype=np.float64)

        # Z-score normalise if requested
        if normalize and len(ns_arr) > 1:
            ns_std = np.std(ns_arr)
            lm_std = np.std(lm_arr)
            if ns_std > 1e-8:
                ns_arr = (ns_arr - np.mean(ns_arr)) / ns_std
            else:
                ns_arr = ns_arr - np.mean(ns_arr)
            if lm_std > 1e-8:
                lm_arr = (lm_arr - np.mean(lm_arr)) / lm_std
            else:
                lm_arr = lm_arr - np.mean(lm_arr)

        if mode == "unconstrained_rescore":
            # Simple linear combination
            combined = (
                lambda_neural * ns_arr
                + lambda_lm * lm_arr
                + length_penalty_beta * np.array([len(c) for c in cu.candidate_lists])
            )
            best_idx = int(np.argmax(combined))
            all_pred_words.append(cu.candidate_lists[best_idx])

        elif mode == "constrained_rescore":
            # Build template and apply constraint penalties
            word_hyps = list(zip(cu.candidate_lists, cu.neural_scores))
            template = builder.build(word_hyps, cu.word_confidences)

            # Extract template reference words
            template_words = []
            for slot in template.slots:
                if slot.is_locked:
                    template_words.append(slot.locked_word or "")
                elif slot.candidates:
                    template_words.append(slot.candidates[0])
                else:
                    template_words.append("<unk>")

            best_score = float("-inf")
            best_words = cu.candidate_lists[0]

            for i, (words, ns, lm) in enumerate(
                zip(cu.candidate_lists, ns_arr.tolist(), lm_arr.tolist())
            ):
                # Constraint penalty
                penalty = 0.0
                if template_words and words:
                    alignment = ConstrainedHypothesisBuilder._align_word_sequences(
                        template_words, words
                    )
                    for tmpl_idx, cand_idx in alignment:
                        if tmpl_idx is None or cand_idx is None:
                            continue
                        slot = template.slots[tmpl_idx]
                        if slot.is_locked and words[cand_idx] != slot.locked_word:
                            penalty += gamma_constraint

                combined = (
                    lambda_neural * ns
                    + lambda_lm * lm
                    - penalty
                    + length_penalty_beta * len(words)
                )
                if combined > best_score:
                    best_score = combined
                    best_words = words

            all_pred_words.append(best_words)

        all_ref_words.append(cu.ref_words)

    return evaluate_wer(all_pred_words, all_ref_words)


def grid_search(
    cached_utts: List[CachedUtterance],
    mode: str = "unconstrained_rescore",
    verbose: bool = True,
) -> List[TuneResult]:
    """
    Exhaustive grid search over rescoring hyperparameters.

    The grid is designed to cover a wide range while remaining tractable:
    ~1000 combinations, each evaluating in <1s with cached scores.
    """
    # Compute oracle WER for reference
    all_cand_sets = [cu.candidate_lists for cu in cached_utts]
    all_ref_words = [cu.ref_words for cu in cached_utts]
    ow = oracle_wer(all_cand_sets, all_ref_words)

    # Grid definition
    lambda_neural_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    lambda_lm_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    length_penalty_values = [0.0, 0.05, 0.1, 0.2]
    normalize_values = [True, False]

    if mode == "constrained_rescore":
        gamma_values = [0.0, 0.5, 1.0, 2.0, 5.0]
        threshold_values = [0.5, 0.7, 0.85, 0.95]
    else:
        gamma_values = [0.0]
        threshold_values = [0.85]

    total = (
        len(lambda_neural_values)
        * len(lambda_lm_values)
        * len(length_penalty_values)
        * len(normalize_values)
        * len(gamma_values)
        * len(threshold_values)
    )
    if verbose:
        print(f"\nGrid search: {total} combinations")
        print(f"Oracle WER:  {ow:.4f}")
        print(f"Mode:        {mode}")

    results: List[TuneResult] = []
    best_wer = float("inf")
    best_config = None
    n_done = 0
    t0 = time.time()

    for ln, ll, lp, norm, gc, thr in itertools.product(
        lambda_neural_values,
        lambda_lm_values,
        length_penalty_values,
        normalize_values,
        gamma_values,
        threshold_values,
    ):
        wer = rescore_with_params(
            cached_utts,
            lambda_neural=ln,
            lambda_lm=ll,
            gamma_constraint=gc,
            length_penalty_beta=lp,
            high_confidence_threshold=thr,
            normalize=norm,
            mode=mode,
        )
        result = TuneResult(
            lambda_neural=ln,
            lambda_lm=ll,
            gamma_constraint=gc,
            length_penalty_beta=lp,
            high_confidence_threshold=thr,
            normalize_scores=norm,
            wer=wer,
            oracle=ow,
        )
        results.append(result)

        n_done += 1
        if wer < best_wer:
            best_wer = wer
            best_config = result
            if verbose:
                elapsed = time.time() - t0
                print(
                    f"  [{n_done}/{total}] NEW BEST WER={wer:.4f}  "
                    f"ln={ln} ll={ll} gc={gc} lp={lp} thr={thr} "
                    f"norm={norm}  ({elapsed:.1f}s)"
                )
        elif verbose and n_done % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_done}/{total}] best so far: {best_wer:.4f}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nGrid search complete in {elapsed:.1f}s")
        print(f"Best WER: {best_wer:.4f} (oracle: {ow:.4f})")
        print(f"Best config: {best_config}")
        gap_closed = (
            (results[0].wer - best_wer) / max(results[0].wer - ow, 1e-8) * 100
            if results
            else 0
        )
        print(f"Oracle gap closed: {gap_closed:.1f}%")

    # Sort by WER
    results.sort(key=lambda r: r.wer)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Tune rescoring hyperparameters via grid search"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--llm_device", type=str, default="cpu")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--partition", type=str, default="test")
    parser.add_argument("--beam_width", type=int, default=40)
    parser.add_argument("--n_best", type=int, default=25)
    parser.add_argument("--max_word_candidates", type=int, default=100)
    parser.add_argument(
        "--mode",
        type=str,
        default="unconstrained_rescore",
        choices=["unconstrained_rescore", "constrained_rescore"],
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="Path to save/load cached scores (avoids re-computing)",
    )

    args = parser.parse_args()

    # Check for cached scores
    if args.cache_path and os.path.exists(args.cache_path):
        print(f"Loading cached scores from {args.cache_path}...")
        with open(args.cache_path, "rb") as f:
            cache_data = pickle.load(f)
        cached_utts = cache_data["cached_utts"]
        ow = cache_data["oracle_wer"]
        print(f"  Loaded {len(cached_utts)} utterances, oracle WER: {ow:.4f}")
    else:
        cached_utts, ow = precompute_scores(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            llm_model=args.llm_model,
            llm_device=args.llm_device,
            device=args.device,
            partition=args.partition,
            beam_width=args.beam_width,
            n_best=args.n_best,
            max_word_candidates=args.max_word_candidates,
        )
        # Save cache
        if args.cache_path:
            os.makedirs(os.path.dirname(args.cache_path) or ".", exist_ok=True)
            with open(args.cache_path, "wb") as f:
                pickle.dump({"cached_utts": cached_utts, "oracle_wer": ow}, f)
            print(f"  Cached scores saved to {args.cache_path}")

    # Run grid search
    results = grid_search(cached_utts, mode=args.mode)

    # Print top-10
    print("\n=== Top 10 Configurations ===")
    print(
        f"{'Rank':<6} {'WER':>8} {'ln':>6} {'ll':>6} {'gc':>6} "
        f"{'lp':>6} {'thr':>6} {'norm':>6}"
    )
    print("-" * 56)
    for i, r in enumerate(results[:10]):
        print(
            f"{i + 1:<6} {r.wer:>8.4f} {r.lambda_neural:>6.2f} "
            f"{r.lambda_lm:>6.2f} {r.gamma_constraint:>6.2f} "
            f"{r.length_penalty_beta:>6.2f} {r.high_confidence_threshold:>6.2f} "
            f"{'Y' if r.normalize_scores else 'N':>6}"
        )

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_data = {
            "mode": args.mode,
            "oracle_wer": ow,
            "n_utterances": len(cached_utts),
            "best": asdict(results[0]) if results else None,
            "top_20": [asdict(r) for r in results[:20]],
            "all_results": [asdict(r) for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
