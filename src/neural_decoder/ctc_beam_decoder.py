"""
A1 – CTC Beam Search Decoder
=============================
Pure-Python prefix beam search that produces an N-best list of phoneme
sequences together with their CTC log-probabilities and frame-level
alignment information.

The implementation follows the classic CTC prefix beam search algorithm
(Hannun 2017) with extensions for:
  * N-best extraction (keep top-K completed prefixes)
  * CTC forced alignment (Viterbi constrained to hypothesis sequence)
  * Forward-backward posteriors for uncertainty over aligned spans
  * Optional blank penalty to control insertion/deletion balance

Usage
-----
    from neural_decoder.ctc_beam_decoder import CTCBeamDecoder
    decoder = CTCBeamDecoder(beam_width=25, blank=0)
    results = decoder.decode(log_probs, length)
    # results is a list of CTCHypothesis
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CTCHypothesis:
    """One entry in the N-best list produced by beam search."""

    phoneme_ids: List[int]  # decoded phoneme sequence (no blanks/repeats)
    log_prob: float  # total CTC log-probability
    frame_alignment: List[int]  # per-frame forced alignment (Viterbi constrained)
    frame_log_probs: np.ndarray  # [T, C] log posteriors used for this hyp
    # Per-label forward-backward posteriors: gamma[t, s] = P(label s active at t | X, Y)
    # Shape [T, S] where S = len(expanded_labels) = 2*len(phoneme_ids)+1
    label_posteriors: Optional[np.ndarray] = None


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


# ======================================================================
# CTC Forced Alignment (Viterbi + Forward-Backward)
# ======================================================================


def ctc_forced_align(
    log_probs: np.ndarray,
    label_seq: List[int],
    blank: int = 0,
) -> Tuple[List[int], np.ndarray]:
    """
    CTC forced alignment via Viterbi decoding constrained to *label_seq*.

    Constructs the standard CTC expanded label sequence (with blanks
    interleaved) and runs Viterbi to find the best frame-to-label
    assignment consistent with *label_seq*.

    Parameters
    ----------
    log_probs : np.ndarray, shape [T, C]
        Frame-level log-softmax posteriors.
    label_seq : list of int
        Hypothesis phoneme sequence (no blanks, no repeats).
    blank : int
        Blank index.

    Returns
    -------
    alignment : list of int, length T
        Per-frame label assignment from the expanded label set.
        Values are label IDs (including blank).
    expanded_labels : np.ndarray
        The expanded label sequence [blank, l0, blank, l1, ..., blank].
    """
    if not label_seq:
        return [blank] * log_probs.shape[0], np.array([blank])

    T, C = log_probs.shape
    L = len(label_seq)
    S = 2 * L + 1  # expanded label length

    # Build expanded label sequence: blank l0 blank l1 ... blank
    expanded = np.zeros(S, dtype=np.int64)
    for i in range(L):
        expanded[2 * i] = blank
        expanded[2 * i + 1] = label_seq[i]
    expanded[2 * L] = blank

    NEG_INF = -1e30

    # Viterbi trellis: alpha[t, s] = best log-prob reaching state s at time t
    alpha = np.full((T, S), NEG_INF, dtype=np.float64)
    # Backpointer
    bp = np.zeros((T, S), dtype=np.int64)

    # Initialise t=0: can only start at state 0 (blank) or state 1 (first label)
    alpha[0, 0] = log_probs[0, expanded[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, expanded[1]]

    for t in range(1, T):
        for s in range(S):
            emit = log_probs[t, expanded[s]]

            # Stay in same state
            best = alpha[t - 1, s]
            best_s = s

            # Transition from previous state
            if s >= 1 and alpha[t - 1, s - 1] > best:
                best = alpha[t - 1, s - 1]
                best_s = s - 1

            # Skip-transition (non-blank to non-blank, skipping a blank,
            # only if the two non-blank labels are different)
            if (
                s >= 2
                and expanded[s] != blank
                and expanded[s] != expanded[s - 2]
                and alpha[t - 1, s - 2] > best
            ):
                best = alpha[t - 1, s - 2]
                best_s = s - 2

            alpha[t, s] = best + emit
            bp[t, s] = best_s

    # Backtrace: final state is either S-1 (last blank) or S-2 (last label)
    if S >= 2 and alpha[T - 1, S - 2] > alpha[T - 1, S - 1]:
        s = S - 2
    else:
        s = S - 1

    alignment = [0] * T
    alignment[T - 1] = int(expanded[s])
    for t in range(T - 2, -1, -1):
        s = bp[t + 1, s]
        alignment[t] = int(expanded[s])

    return alignment, expanded


def ctc_forward_backward(
    log_probs: np.ndarray,
    label_seq: List[int],
    blank: int = 0,
) -> np.ndarray:
    """
    CTC forward-backward algorithm for computing per-frame label posteriors.

    Returns gamma[t, s] = P(state s at time t | X, label_seq), where s
    indexes into the expanded label sequence.

    Parameters
    ----------
    log_probs : np.ndarray, shape [T, C]
    label_seq : list of int
    blank : int

    Returns
    -------
    gamma : np.ndarray, shape [T, S]
        Posterior probabilities (in probability space, not log).
    """
    if not label_seq:
        T = log_probs.shape[0]
        return np.ones((T, 1), dtype=np.float64)

    T, C = log_probs.shape
    L = len(label_seq)
    S = 2 * L + 1

    expanded = np.zeros(S, dtype=np.int64)
    for i in range(L):
        expanded[2 * i] = blank
        expanded[2 * i + 1] = label_seq[i]
    expanded[2 * L] = blank

    NEG_INF = -1e30

    # Forward pass
    fwd = np.full((T, S), NEG_INF, dtype=np.float64)
    fwd[0, 0] = log_probs[0, expanded[0]]
    if S > 1:
        fwd[0, 1] = log_probs[0, expanded[1]]

    for t in range(1, T):
        for s in range(S):
            emit = log_probs[t, expanded[s]]
            # Same state
            val = fwd[t - 1, s]
            # Previous state
            if s >= 1:
                val = _log_add(val, fwd[t - 1, s - 1])
            # Skip (non-blank to different non-blank over blank)
            if s >= 2 and expanded[s] != blank and expanded[s] != expanded[s - 2]:
                val = _log_add(val, fwd[t - 1, s - 2])
            fwd[t, s] = val + emit

    # Backward pass
    bwd = np.full((T, S), NEG_INF, dtype=np.float64)
    bwd[T - 1, S - 1] = 0.0
    if S >= 2:
        bwd[T - 1, S - 2] = 0.0

    for t in range(T - 2, -1, -1):
        for s in range(S):
            val = bwd[t + 1, s] + log_probs[t + 1, expanded[s]]
            if s + 1 < S:
                val = _log_add(
                    val, bwd[t + 1, s + 1] + log_probs[t + 1, expanded[s + 1]]
                )
            if (
                s + 2 < S
                and expanded[s + 2] != blank
                and expanded[s + 2] != expanded[s]
            ):
                val = _log_add(
                    val, bwd[t + 1, s + 2] + log_probs[t + 1, expanded[s + 2]]
                )
            bwd[t, s] = val

    # Combine: log gamma[t,s] = fwd[t,s] + bwd[t,s]
    log_gamma = fwd + bwd
    # Total log-likelihood
    log_Z = _log_add(fwd[T - 1, S - 1], fwd[T - 1, S - 2] if S >= 2 else NEG_INF)

    # Normalise
    log_gamma -= log_Z
    gamma = np.exp(np.clip(log_gamma, -500, 0))

    return gamma


class CTCBeamDecoder:
    """
    CTC prefix beam search decoder.

    Parameters
    ----------
    beam_width : int
        Number of active prefixes to keep at each time-step.
    blank : int
        Index of the CTC blank token (default 0).
    n_best : int
        How many top hypotheses to return.
    blank_penalty : float
        Additive log-penalty for blank emissions (positive = penalise blanks,
        which encourages more non-blank emissions).
    """

    def __init__(
        self,
        beam_width: int = 25,
        blank: int = 0,
        n_best: int = 10,
        blank_penalty: float = 0.0,
    ):
        self.beam_width = beam_width
        self.blank = blank
        self.n_best = n_best
        self.blank_penalty = blank_penalty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None,
    ) -> List[CTCHypothesis]:
        """
        Run CTC prefix beam search on a single utterance.

        Parameters
        ----------
        log_probs : np.ndarray, shape [T, C]
            Frame-level log-softmax outputs from the acoustic model.
        length : int or None
            Actual number of valid frames (if padded).  Uses all frames
            when *None*.

        Returns
        -------
        List[CTCHypothesis]
            Up to *n_best* hypotheses sorted best-first.
        """
        T, C = log_probs.shape
        if length is not None:
            T = min(T, int(length))
        log_probs = log_probs[:T]

        # Apply blank penalty
        penalised = log_probs.copy()
        if self.blank_penalty != 0.0:
            penalised[:, self.blank] -= self.blank_penalty

        # ---- prefix beam search ----
        # State: dict  prefix_tuple -> (p_blank, p_non_blank)
        # where values are log-probabilities
        NEG_INF = -math.inf
        beams: Dict[Tuple[int, ...], List[float]] = {
            (): [penalised[0, self.blank], NEG_INF]
        }
        # Initialise with first frame non-blank emissions
        for c in range(C):
            if c == self.blank:
                continue
            beams[(c,)] = [NEG_INF, penalised[0, c]]

        for t in range(1, T):
            new_beams: Dict[Tuple[int, ...], List[float]] = {}

            # Prune to beam_width
            scored = {prefix: _log_add(pb, pnb) for prefix, (pb, pnb) in beams.items()}
            top_prefixes = sorted(scored, key=lambda k: scored[k], reverse=True)[
                : self.beam_width
            ]

            for prefix in top_prefixes:
                pb, pnb = beams[prefix]
                p_total = _log_add(pb, pnb)

                # --- extend with blank ---
                new_pb = p_total + penalised[t, self.blank]
                if prefix not in new_beams:
                    new_beams[prefix] = [NEG_INF, NEG_INF]
                new_beams[prefix][0] = _log_add(new_beams[prefix][0], new_pb)

                # --- extend with non-blank ---
                for c in range(C):
                    if c == self.blank:
                        continue
                    ext = prefix + (c,)
                    if prefix and c == prefix[-1]:
                        # Same label as last – only the blank-ending path
                        # may emit this label without collapsing
                        new_pnb = pb + penalised[t, c]
                        if ext not in new_beams:
                            new_beams[ext] = [NEG_INF, NEG_INF]
                        new_beams[ext][1] = _log_add(new_beams[ext][1], new_pnb)
                        # Also continue the current prefix with the repeat
                        # collapsed (via non-blank path)
                        if prefix not in new_beams:
                            new_beams[prefix] = [NEG_INF, NEG_INF]
                        new_beams[prefix][1] = _log_add(
                            new_beams[prefix][1], pnb + penalised[t, c]
                        )
                    else:
                        new_pnb = p_total + penalised[t, c]
                        if ext not in new_beams:
                            new_beams[ext] = [NEG_INF, NEG_INF]
                        new_beams[ext][1] = _log_add(new_beams[ext][1], new_pnb)

            beams = new_beams

        # ---- collect N-best ----
        scored_final = {
            prefix: _log_add(pb, pnb) for prefix, (pb, pnb) in beams.items()
        }
        sorted_prefixes = sorted(
            scored_final, key=lambda k: scored_final[k], reverse=True
        )

        results: List[CTCHypothesis] = []
        for prefix in sorted_prefixes[: self.n_best]:
            phone_ids = list(prefix)

            # CTC forced alignment constrained to this hypothesis
            if phone_ids:
                alignment, _ = ctc_forced_align(log_probs, phone_ids, blank=self.blank)
                # Forward-backward posteriors for uncertainty
                gamma = ctc_forward_backward(log_probs, phone_ids, blank=self.blank)
            else:
                alignment = [self.blank] * T
                gamma = np.ones((T, 1), dtype=np.float64)

            results.append(
                CTCHypothesis(
                    phoneme_ids=phone_ids,
                    log_prob=scored_final[prefix],
                    frame_alignment=alignment,
                    frame_log_probs=log_probs,
                    label_posteriors=gamma,
                )
            )

        return results

    def decode_batch(
        self,
        log_probs_batch: np.ndarray,
        lengths: Optional[np.ndarray] = None,
    ) -> List[List[CTCHypothesis]]:
        """
        Decode a batch of utterances.

        Parameters
        ----------
        log_probs_batch : np.ndarray, shape [B, T, C]
        lengths : np.ndarray or None, shape [B]

        Returns
        -------
        List of N-best lists, one per utterance.
        """
        B = log_probs_batch.shape[0]
        results = []
        for b in range(B):
            length = int(lengths[b]) if lengths is not None else None
            results.append(self.decode(log_probs_batch[b], length))
        return results
