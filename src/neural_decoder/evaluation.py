"""
E – Evaluation Pipeline
========================
Computes WER, CER, PER, constraint adherence, runtime, and
confidence-bucket breakdowns for the full ablation ladder:

1. Neural-only (greedy CTC)
2. + CTC beam search
3. + Lexicon phoneme→word
4. + LLM rescoring (unconstrained)
5. + Posterior-constrained LLM (novelty)
6. + Distilled student

Usage
-----
    from neural_decoder.evaluation import evaluate_wer, evaluate_cer, EvalReport
    wer = evaluate_wer(predicted_words, reference_words)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from edit_distance import SequenceMatcher


# ======================================================================
# Core metrics
# ======================================================================


def edit_distance(a: List, b: List) -> int:
    """Levenshtein edit distance between two sequences."""
    matcher = SequenceMatcher(a=a, b=b)
    return matcher.distance()


def evaluate_wer(
    predictions: List[List[str]],
    references: List[List[str]],
) -> float:
    """
    Word Error Rate (WER) = total_edits / total_ref_words.

    Parameters
    ----------
    predictions : list of word lists
    references : list of word lists

    Returns
    -------
    float  WER in [0, ∞)
    """
    total_edits = 0
    total_ref = 0
    for pred, ref in zip(predictions, references):
        total_edits += edit_distance(ref, pred)
        total_ref += len(ref)
    return total_edits / max(total_ref, 1)


def evaluate_cer(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Character Error Rate (CER) = total_char_edits / total_ref_chars.

    Parameters
    ----------
    predictions : list of strings
    references : list of strings
    """
    total_edits = 0
    total_ref = 0
    for pred, ref in zip(predictions, references):
        total_edits += edit_distance(list(ref), list(pred))
        total_ref += len(ref)
    return total_edits / max(total_ref, 1)


def evaluate_per(
    predictions: List[List[int]],
    references: List[List[int]],
) -> float:
    """
    Phoneme Error Rate (PER) = total_phone_edits / total_ref_phones.
    """
    total_edits = 0
    total_ref = 0
    for pred, ref in zip(predictions, references):
        total_edits += edit_distance(ref, pred)
        total_ref += len(ref)
    return total_edits / max(total_ref, 1)


def constraint_adherence(
    predictions: List[List[str]],
    candidate_sets: List[List[List[str]]],
) -> float:
    """
    Word-level constraint adherence: fraction of predicted words that
    appear in the candidate set at any position.

    For **rescoring** mode: adherence is 100% by construction since the
    output is always one of the N-best candidates verbatim.

    For **slot-filling** mode: each word at each position comes from the
    confusion set for that slot, so adherence is 100% by construction.

    The metric is most useful as a sanity check — any deviation from
    1.0 indicates a bug in the constrained decoding pipeline.
    """
    total_words = 0
    in_set = 0
    for pred_words, cand_set in zip(predictions, candidate_sets):
        all_cand_words = set()
        for cand in cand_set:
            for w in cand:
                all_cand_words.add(w)
        for w in pred_words:
            total_words += 1
            if w in all_cand_words:
                in_set += 1
    return in_set / max(total_words, 1)


def transcript_adherence(
    predictions: List[List[str]],
    candidate_sets: List[List[List[str]]],
) -> float:
    """
    Transcript-level adherence: fraction of predicted transcripts that
    exactly match one of the N-best candidates.

    This is a stricter metric than word-level adherence. For rescoring
    mode, it should be 100% (output is always an N-best candidate).
    For slot-filling mode, it may be < 100% since the output is
    constructed word-by-word from confusion sets and may not match any
    single N-best candidate exactly.

    Returns
    -------
    float in [0, 1]
    """
    if not predictions:
        return 1.0

    n_match = 0
    for pred_words, cand_set in zip(predictions, candidate_sets):
        pred_tuple = tuple(pred_words)
        if any(tuple(cand) == pred_tuple for cand in cand_set):
            n_match += 1
    return n_match / len(predictions)


def slot_adherence(
    predictions: List[List[str]],
    slot_candidate_sets: List[List[List[str]]],
) -> float:
    """
    Slot-level adherence for slot-filling mode: for each position,
    check that the predicted word is in the confusion set for that slot.

    Parameters
    ----------
    predictions : list of word lists
    slot_candidate_sets : list of per-utterance per-slot candidate lists
        slot_candidate_sets[i][j] = list of candidate words for position j

    Returns
    -------
    float in [0, 1]
    """
    total_slots = 0
    in_set = 0
    for pred_words, per_slot_cands in zip(predictions, slot_candidate_sets):
        for j, w in enumerate(pred_words):
            total_slots += 1
            if j < len(per_slot_cands) and w in per_slot_cands[j]:
                in_set += 1
    return in_set / max(total_slots, 1)


# ======================================================================
# Confidence-bucket analysis
# ======================================================================


def wer_by_confidence_bucket(
    word_predictions: List[str],
    word_references: List[str],
    word_confidences: List[float],
    n_buckets: int = 5,
) -> List[Tuple[float, float, float, int]]:
    """
    Break down WER improvement by neural confidence buckets.

    Returns list of (bucket_low, bucket_high, bucket_wer, n_words).
    """
    if not word_confidences:
        return []

    confs = np.array(word_confidences)
    bucket_edges = np.linspace(0, 1, n_buckets + 1)

    results: List[Tuple[float, float, float, int]] = []
    for i in range(n_buckets):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (confs >= lo) & (confs < hi if i < n_buckets - 1 else confs <= hi)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            results.append((lo, hi, 0.0, 0))
            continue

        bucket_preds = [word_predictions[j] for j in indices]
        bucket_refs = [word_references[j] for j in indices]

        edits = sum(1 for p, r in zip(bucket_preds, bucket_refs) if p != r)
        bucket_wer = edits / max(len(bucket_refs), 1)
        results.append((lo, hi, bucket_wer, len(indices)))

    return results


# ======================================================================
# Evaluation report
# ======================================================================


@dataclass
class EvalReport:
    """Structured evaluation report for one decoding configuration."""

    name: str = ""
    wer: float = 0.0
    cer: float = 0.0
    per: float = 0.0
    constraint_adherence: float = 0.0
    transcript_adherence: float = 0.0  # entire output matches one N-best candidate
    slot_adherence: float = 0.0  # per-slot word from confusion set (slot-filling only)
    hallucination_rate_word: float = 0.0  # 1 - word-level constraint adherence
    hallucination_rate_transcript: float = 0.0  # 1 - transcript adherence
    runtime_seconds: float = 0.0
    n_parameters: int = 0
    confidence_breakdown: List[Tuple[float, float, float, int]] = field(
        default_factory=list
    )
    extra: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== {self.name} ===",
            f"  WER:                       {self.wer:.4f}",
            f"  CER:                       {self.cer:.4f}",
            f"  PER:                       {self.per:.4f}",
            f"  Constraint adherence (word):       {self.constraint_adherence:.4f}",
            f"  Constraint adherence (transcript): {self.transcript_adherence:.4f}",
            f"  Slot adherence:                    {self.slot_adherence:.4f}",
            f"  Hallucination rate (word):         {self.hallucination_rate_word:.4f}",
            f"  Hallucination rate (transcript):   {self.hallucination_rate_transcript:.4f}",
            f"  Runtime (s):               {self.runtime_seconds:.2f}",
            f"  Parameters:                {self.n_parameters:,}",
        ]
        if self.confidence_breakdown:
            lines.append("  Confidence buckets:")
            for lo, hi, bwer, n in self.confidence_breakdown:
                lines.append(f"    [{lo:.2f}, {hi:.2f}): WER={bwer:.4f}  n={n}")
        for k, v in self.extra.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def run_ablation_ladder(
    reports: List[EvalReport],
) -> str:
    """
    Format a comparison table across multiple decoding configurations.
    """
    header = (
        f"{'Method':<35} {'WER':>8} {'CER':>8} {'PER':>8} "
        f"{'C-Adh':>8} {'T-Adh':>8} {'H-Rate':>8} "
        f"{'Time(s)':>10} {'Params':>12}"
    )
    lines = [header, "-" * len(header)]
    for r in reports:
        lines.append(
            f"{r.name:<35} {r.wer:>8.4f} {r.cer:>8.4f} {r.per:>8.4f} "
            f"{r.constraint_adherence:>8.4f} {r.transcript_adherence:>8.4f} "
            f"{r.hallucination_rate_transcript:>8.4f} "
            f"{r.runtime_seconds:>10.2f} {r.n_parameters:>12,}"
        )
    return "\n".join(lines)
