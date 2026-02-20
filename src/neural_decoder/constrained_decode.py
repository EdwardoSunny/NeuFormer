"""
B2 + C2 – Constrained Hypothesis Space & Slot-Filling Decoder
==============================================================
Builds a *posterior-constrained* hypothesis space from the N-best
word-level candidates and word-level confidence scores, then
provides two modes of LLM-assisted decoding:

1. **Constrained N-best rescoring** (C1-style): only compare
   candidates that respect high-confidence locked spans.
2. **Slot-filling mode** (C2): build a template with locked spans
   and confusion-set slots, then search over slot combinations
   using LLM incremental scoring.

Usage
-----
    from neural_decoder.constrained_decode import (
        ConstrainedHypothesisBuilder, SlotFillingDecoder
    )
    builder = ConstrainedHypothesisBuilder()
    template = builder.build(word_hypotheses, word_confidences)
    decoder = SlotFillingDecoder(llm_scorer)
    best = decoder.decode(template)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class SlotEntry:
    """One slot (position) in the decoding template."""

    position: int  # word position in the sequence
    is_locked: bool  # True = high-confidence, fixed
    locked_word: Optional[str] = None
    candidates: List[str] = field(default_factory=list)
    neural_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class ConstrainedTemplate:
    """
    A constrained decoding template for one utterance.

    Contains a sequence of slots, each either locked (high-confidence)
    or open (with a confusion set of alternatives).
    """

    slots: List[SlotEntry] = field(default_factory=list)
    neural_score: float = 0.0  # overall neural-model score

    @property
    def n_open_slots(self) -> int:
        return sum(1 for s in self.slots if not s.is_locked)

    @property
    def n_locked_slots(self) -> int:
        return sum(1 for s in self.slots if s.is_locked)

    def get_best_candidate_sentence(self) -> str:
        """Return the sentence formed by picking the top candidate per slot."""
        words = []
        for slot in self.slots:
            if slot.is_locked:
                words.append(slot.locked_word or "")
            elif slot.candidates:
                words.append(slot.candidates[0])
            else:
                words.append("<unk>")
        return " ".join(words)

    def enumerate_candidates(self, max_combinations: int = 1000) -> List[List[str]]:
        """
        Enumerate candidate word sequences up to max_combinations.

        For locked slots the word is fixed.  For open slots, iterate
        over the candidate set.  Uses iterative deepening to avoid
        combinatorial explosion.
        """
        # Build per-slot candidate lists
        per_slot: List[List[str]] = []
        for slot in self.slots:
            if slot.is_locked:
                per_slot.append([slot.locked_word or ""])
            else:
                per_slot.append(slot.candidates if slot.candidates else ["<unk>"])

        # Enumerate via generator with early cutoff
        results: List[List[str]] = []
        self._enumerate_recursive(per_slot, 0, [], results, max_combinations)
        return results

    def _enumerate_recursive(
        self,
        per_slot: List[List[str]],
        idx: int,
        current: List[str],
        results: List[List[str]],
        max_results: int,
    ):
        if len(results) >= max_results:
            return
        if idx == len(per_slot):
            results.append(list(current))
            return
        for word in per_slot[idx]:
            current.append(word)
            self._enumerate_recursive(per_slot, idx + 1, current, results, max_results)
            current.pop()


# ======================================================================
# B2 – Constrained Hypothesis Builder
# ======================================================================


class ConstrainedHypothesisBuilder:
    """
    Builds a ConstrainedTemplate from word-level N-best hypotheses
    and per-word confidence scores.

    Parameters
    ----------
    high_confidence_threshold : float
        Words with confidence >= this threshold are locked.
    max_candidates_per_slot : int
        Maximum number of alternative words per open slot.
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.7,
        max_candidates_per_slot: int = 20,
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.max_candidates_per_slot = max_candidates_per_slot

    def build(
        self,
        word_hypotheses: List[Tuple[List[str], float]],
        word_confidences: List[Tuple[str, float]],
    ) -> ConstrainedTemplate:
        """
        Build a constrained template.

        Parameters
        ----------
        word_hypotheses : list of (word_list, neural_score)
            N-best word-level candidates for the utterance.
        word_confidences : list of (word, confidence)
            Per-word confidence from the best (first) hypothesis.
            Length should match the first hypothesis word count.

        Returns
        -------
        ConstrainedTemplate
        """
        if not word_hypotheses:
            return ConstrainedTemplate()

        best_words, best_score = word_hypotheses[0]
        n_words = len(best_words)

        # Build slots
        slots: List[SlotEntry] = []
        for i in range(n_words):
            word = best_words[i]
            conf = word_confidences[i][1] if i < len(word_confidences) else 0.0
            is_locked = conf >= self.high_confidence_threshold

            slot = SlotEntry(
                position=i,
                is_locked=is_locked,
                locked_word=word if is_locked else None,
                candidates=[word] if not is_locked else [],
                confidence=conf,
            )

            if not is_locked:
                slot.neural_scores[word] = best_score

            slots.append(slot)

        # Populate open-slot candidates from N-best hypotheses
        for hyp_words, hyp_score in word_hypotheses[1:]:
            for i, slot in enumerate(slots):
                if slot.is_locked:
                    continue
                if i < len(hyp_words):
                    alt = hyp_words[i]
                    if alt not in slot.candidates:
                        slot.candidates.append(alt)
                        slot.neural_scores[alt] = hyp_score
                    if len(slot.candidates) >= self.max_candidates_per_slot:
                        break

        return ConstrainedTemplate(slots=slots, neural_score=best_score)


# ======================================================================
# C2 – Slot-Filling Decoder
# ======================================================================


class SlotFillingDecoder:
    """
    LLM-guided slot-filling decoder.

    For each open slot in the template, the LLM scores every candidate
    word in context, and beam search picks the globally best combination.

    Parameters
    ----------
    llm_score_fn : callable
        Function (text: str) -> float that returns the length-normalized
        log-probability of the text under the LLM.
    lambda_neural : float
        Weight for the neural (CTC-based) score component.
    lambda_lm : float
        Weight for the LLM score component.
    gamma_constraint : float
        Penalty per edit on high-confidence (locked) slots.
    beam_width : int
        Beam width for slot-filling search.
    """

    def __init__(
        self,
        llm_score_fn: Callable[[str], float],
        lambda_neural: float = 1.0,
        lambda_lm: float = 0.5,
        gamma_constraint: float = 5.0,
        beam_width: int = 10,
    ):
        self.llm_score_fn = llm_score_fn
        self.lambda_neural = lambda_neural
        self.lambda_lm = lambda_lm
        self.gamma_constraint = gamma_constraint
        self.beam_width = beam_width

    def decode(self, template: ConstrainedTemplate) -> Tuple[List[str], float]:
        """
        Run slot-filling decoding on a constrained template.

        Returns
        -------
        (best_word_sequence, combined_score)
        """
        if not template.slots:
            return [], 0.0

        # Beam: list of (words_so_far, accumulated_score)
        beam: List[Tuple[List[str], float]] = [([], 0.0)]

        for slot in template.slots:
            new_beam: List[Tuple[List[str], float]] = []

            if slot.is_locked:
                # Only one choice
                word = slot.locked_word or ""
                for words, score in beam:
                    new_beam.append((words + [word], score))
            else:
                # Try each candidate
                for words, score in beam:
                    for cand in slot.candidates:
                        # Build partial sentence for LLM scoring
                        partial = " ".join(words + [cand])
                        lm_score = self.llm_score_fn(partial)

                        # Neural score for this candidate at this position
                        neural_score = slot.neural_scores.get(cand, -10.0)

                        combined = (
                            score
                            + self.lambda_neural * neural_score
                            + self.lambda_lm * lm_score
                        )
                        new_beam.append((words + [cand], combined))

            # Prune beam
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[: self.beam_width]

        if not beam:
            return [], 0.0

        return beam[0]

    def rescore_nbest(
        self,
        template: ConstrainedTemplate,
        candidates: List[List[str]],
        neural_scores: List[float],
        length_penalty_beta: float = 0.0,
    ) -> List[Tuple[List[str], float]]:
        """
        C1-style constrained N-best rescoring.

        **Constraint adherence**: The output is always one of the input
        candidates verbatim, so transcript-level adherence is 100% by
        construction.

        **Locked-slot penalty**: Although the output is always a
        candidate, the ``gamma_constraint`` penalty still serves a
        purpose: different candidates may violate locked spans at
        different positions, and the penalty acts as a *soft preference*
        for candidates that respect the high-confidence locked words.
        This is not redundant — it biases the rescorer toward
        candidates consistent with the posterior-constrained template.

        Parameters
        ----------
        template : ConstrainedTemplate
        candidates : list of word lists
        neural_scores : parallel list of neural scores
        length_penalty_beta : float
            Length normalisation term.

        Returns
        -------
        list of (word_list, combined_score) sorted best-first.
        """
        results: List[Tuple[List[str], float]] = []

        for words, ns in zip(candidates, neural_scores):
            sentence = " ".join(words)
            lm_score = self.llm_score_fn(sentence)

            # Constraint penalty
            constraint_penalty = 0.0
            for slot in template.slots:
                if slot.is_locked and slot.position < len(words):
                    if words[slot.position] != slot.locked_word:
                        constraint_penalty += self.gamma_constraint

            length_bonus = length_penalty_beta * len(words)

            combined = (
                self.lambda_neural * ns
                + self.lambda_lm * lm_score
                - constraint_penalty
                + length_bonus
            )
            results.append((words, combined))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
