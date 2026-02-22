"""
Evaluation Utilities
=====================
WER computation and result reporting for the decoding pipeline.

Usage
-----
    from neural_decoder.evaluation import compute_wer, remove_punctuation
    wer = compute_wer(predicted_sentences, reference_sentences)
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import editdistance


def remove_punctuation(sentence: str) -> str:
    """
    Remove punctuation from a sentence, keeping only letters, hyphens,
    apostrophes, and spaces.  Lowercases the result.

    Ported from the NEJM brain-to-text repo.
    """
    sentence = re.sub(r"[^a-zA-Z\- ']", "", sentence)
    sentence = sentence.replace("- ", " ").lower()
    sentence = sentence.replace("--", "").lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = " ".join([word for word in sentence.split() if word != ""])
    return sentence


def compute_wer(
    predictions: List[str],
    references: List[str],
    clean: bool = True,
) -> Tuple[float, int, int]:
    """
    Compute aggregate Word Error Rate over a list of sentence pairs.

    Parameters
    ----------
    predictions : list of str
        Predicted sentences.
    references : list of str
        Reference (ground truth) sentences.
    clean : bool
        If True, apply remove_punctuation() to both predictions and references.

    Returns
    -------
    wer : float
        Aggregate WER (total_edit_distance / total_ref_words).
    total_edit_distance : int
        Total word-level edit distance.
    total_ref_words : int
        Total number of reference words.
    """
    total_edit_distance = 0
    total_ref_words = 0

    for pred, ref in zip(predictions, references):
        if clean:
            pred = remove_punctuation(pred)
            ref = remove_punctuation(ref)

        pred_words = pred.strip().split()
        ref_words = ref.strip().split()

        ed = editdistance.eval(ref_words, pred_words)
        total_edit_distance += ed
        total_ref_words += len(ref_words)

    wer = total_edit_distance / max(total_ref_words, 1)
    return wer, total_edit_distance, total_ref_words


def compute_wer_per_sentence(
    predictions: List[str],
    references: List[str],
    clean: bool = True,
) -> List[Tuple[float, int, int]]:
    """
    Compute per-sentence WER.

    Returns
    -------
    list of (wer, edit_distance, n_ref_words)
    """
    results = []
    for pred, ref in zip(predictions, references):
        if clean:
            pred = remove_punctuation(pred)
            ref = remove_punctuation(ref)

        pred_words = pred.strip().split()
        ref_words = ref.strip().split()

        ed = editdistance.eval(ref_words, pred_words)
        n_ref = len(ref_words)
        wer = ed / max(n_ref, 1)
        results.append((wer, ed, n_ref))

    return results


def oracle_wer(
    nbest_lists: List[List[str]],
    references: List[str],
    clean: bool = True,
) -> float:
    """
    Compute oracle WER: for each utterance, pick the n-best candidate
    with the lowest WER against the reference.

    Parameters
    ----------
    nbest_lists : list of list of str
        For each utterance, a list of candidate sentences.
    references : list of str
        Reference sentences.

    Returns
    -------
    float
        Oracle WER.
    """
    total_edit_distance = 0
    total_ref_words = 0

    for candidates, ref in zip(nbest_lists, references):
        if clean:
            ref = remove_punctuation(ref)
        ref_words = ref.strip().split()
        n_ref = len(ref_words)

        best_ed = n_ref  # worst case: all deletions
        for cand in candidates:
            if clean:
                cand = remove_punctuation(cand)
            cand_words = cand.strip().split()
            ed = editdistance.eval(ref_words, cand_words)
            if ed < best_ed:
                best_ed = ed

        total_edit_distance += best_ed
        total_ref_words += n_ref

    return total_edit_distance / max(total_ref_words, 1)
