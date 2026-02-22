"""
N-best Augmentation
====================
Augments the n-best hypothesis list by swapping words between similar
candidates, generating new plausible hypotheses.  Ported directly from
the NEJM brain-to-text repo's ``language-model-standalone.py``.

This is applied between WFST decoding and LLM rescoring to increase
the diversity of the candidate set, which can help the LLM find a
better hypothesis.

Usage
-----
    from neural_decoder.nbest_augmentation import augment_nbest
    augmented = augment_nbest(nbest, top_k=20, acoustic_scale=0.3)
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np


def _get_string_differences(
    cue: str,
    decoder_output: str,
) -> Tuple[int, List, List[Tuple[int, int]]]:
    """
    Compute word-level edit distance and alignment between two sentences.

    Returns
    -------
    cost : int
        Edit distance.
    path : list
        Per-word alignment labels (int index for match, 'I', 'D', 'R').
    indices_to_highlight : list of (start, end)
        Character-level spans of differing words in decoder_output.
    """
    decoder_output_words = decoder_output.split()
    cue_words = cue.split()

    @lru_cache(None)
    def reverse_w_backtrace(i, j):
        if i == 0:
            return j, ["I"] * j
        elif j == 0:
            return i, ["D"] * i
        elif i > 0 and j > 0 and decoder_output_words[i - 1] == cue_words[j - 1]:
            cost, path = reverse_w_backtrace(i - 1, j - 1)
            return cost, path + [i - 1]
        else:
            insertion_cost, insertion_path = reverse_w_backtrace(i, j - 1)
            deletion_cost, deletion_path = reverse_w_backtrace(i - 1, j)
            substitution_cost, substitution_path = reverse_w_backtrace(i - 1, j - 1)
            if insertion_cost <= deletion_cost and insertion_cost <= substitution_cost:
                return insertion_cost + 1, insertion_path + ["I"]
            elif deletion_cost <= insertion_cost and deletion_cost <= substitution_cost:
                return deletion_cost + 1, deletion_path + ["D"]
            else:
                return substitution_cost + 1, substitution_path + ["R"]

    cost, path = reverse_w_backtrace(len(decoder_output_words), len(cue_words))

    # Remove insertions from path
    path = [p for p in path if p != "I"]

    # Get indices in decoder_output of words that differ from cue
    indices_to_highlight = []
    current_index = 0
    for label, word in zip(path, decoder_output_words):
        if label in ["R", "D"]:
            indices_to_highlight.append((current_index, current_index + len(word)))
        current_index += len(word) + 1

    return cost, path, indices_to_highlight


def augment_nbest(
    nbest: List[Tuple[str, float, float]],
    top_candidates_to_augment: int = 20,
    acoustic_scale: float = 0.3,
    score_penalty_percent: float = 0.01,
) -> List[Tuple[str, float, float]]:
    """
    Augment the n-best list by swapping words between top candidates.

    For each pair of top candidates with the same word count, find
    word-level substitutions and create new candidates by swapping
    those words.  New candidates get slightly penalized scores.

    Parameters
    ----------
    nbest : list of (sentence, ac_score, lm_score)
        Original n-best list from the WFST decoder.
    top_candidates_to_augment : int
        Number of top candidates to consider for augmentation.
    acoustic_scale : float
        Acoustic scale for total score computation.
    score_penalty_percent : float
        Percentage penalty applied to augmented candidates' scores.

    Returns
    -------
    list of (sentence, ac_score, lm_score)
        Augmented n-best list, sorted by total score.
    """
    sentences = []
    ac_scores = []
    lm_scores = []
    total_scores = []

    for i in range(len(nbest)):
        sentences.append(nbest[i][0].strip())
        ac_scores.append(nbest[i][1])
        lm_scores.append(nbest[i][2])
        total_scores.append(acoustic_scale * nbest[i][1] + nbest[i][2])

    # Sort by total score
    sorted_indices = np.argsort(total_scores)[::-1]
    sentences = [sentences[i] for i in sorted_indices]
    ac_scores = [ac_scores[i] for i in sorted_indices]
    lm_scores = [lm_scores[i] for i in sorted_indices]
    total_scores = [total_scores[i] for i in sorted_indices]

    # New sentences and scores from word swaps
    new_sentences = []
    new_ac_scores = []
    new_lm_scores = []
    new_total_scores = []

    n_top = min(len(sentences), top_candidates_to_augment)

    for i1 in range(n_top - 1):
        words1 = sentences[i1].split()

        for i2 in range(i1 + 1, n_top):
            words2 = sentences[i2].split()

            if len(words1) != len(words2):
                continue

            _, path1, _ = _get_string_differences(sentences[i1], sentences[i2])
            _, path2, _ = _get_string_differences(sentences[i2], sentences[i1])

            replace_indices1 = [i for i, p in enumerate(path2) if p == "R"]
            replace_indices2 = [i for i, p in enumerate(path1) if p == "R"]

            for r1, r2 in zip(replace_indices1, replace_indices2):
                new_words1 = words1.copy()
                new_words2 = words2.copy()

                new_words1[r1] = words2[r2]
                new_words2[r2] = words1[r1]

                new_sentence1 = " ".join(new_words1)
                new_sentence2 = " ".join(new_words2)

                if (
                    new_sentence1 not in sentences
                    and new_sentence1 not in new_sentences
                ):
                    avg_ac = np.mean([ac_scores[i1], ac_scores[i2]])
                    avg_lm = np.mean([lm_scores[i1], lm_scores[i2]])
                    new_sentences.append(new_sentence1)
                    new_ac_scores.append(avg_ac - score_penalty_percent * abs(avg_ac))
                    new_lm_scores.append(avg_lm - score_penalty_percent * abs(avg_lm))
                    new_total_scores.append(
                        acoustic_scale * new_ac_scores[-1] + new_lm_scores[-1]
                    )

                if (
                    new_sentence2 not in sentences
                    and new_sentence2 not in new_sentences
                ):
                    avg_ac = np.mean([ac_scores[i1], ac_scores[i2]])
                    avg_lm = np.mean([lm_scores[i1], lm_scores[i2]])
                    new_sentences.append(new_sentence2)
                    new_ac_scores.append(avg_ac - score_penalty_percent * abs(avg_ac))
                    new_lm_scores.append(avg_lm - score_penalty_percent * abs(avg_lm))
                    new_total_scores.append(
                        acoustic_scale * new_ac_scores[-1] + new_lm_scores[-1]
                    )

    # Combine with original
    sentences.extend(new_sentences)
    ac_scores.extend(new_ac_scores)
    lm_scores.extend(new_lm_scores)
    total_scores.extend(new_total_scores)

    # Sort by total score
    sorted_indices = np.argsort(total_scores)[::-1]
    nbest_out = []
    for i in sorted_indices:
        nbest_out.append((sentences[i], ac_scores[i], lm_scores[i]))

    return nbest_out
