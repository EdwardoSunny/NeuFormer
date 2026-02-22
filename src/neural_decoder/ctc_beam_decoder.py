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
        for rank, prefix in enumerate(sorted_prefixes[: self.n_best]):
            phone_ids = list(prefix)

            # Only run expensive forced alignment + forward-backward
            # on the 1-best hypothesis (used for uncertainty estimation).
            # Other hypotheses get a cheap argmax alignment.
            if rank == 0 and phone_ids:
                alignment, _ = ctc_forced_align(log_probs, phone_ids, blank=self.blank)
                gamma = ctc_forward_backward(log_probs, phone_ids, blank=self.blank)
            elif phone_ids:
                alignment = np.argmax(log_probs, axis=-1).tolist()
                gamma = None
            else:
                alignment = [self.blank] * T
                gamma = None

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


# ======================================================================
# Word-Level CTC Beam Search (with N-gram shallow fusion)
# ======================================================================


@dataclass
class WordCTCHypothesis:
    """One entry in the word-level N-best list."""

    words: List[str]  # decoded word sequence
    phoneme_ids: List[int]  # full phoneme sequence (with SIL boundaries)
    ctc_log_prob: float  # CTC log-probability of the phoneme sequence
    ngram_log_prob: float  # accumulated N-gram log10 score
    combined_score: float  # alpha * ctc + ngram_alpha * ngram
    frame_alignment: List[int]  # per-frame forced alignment (1-best only)
    frame_log_probs: np.ndarray  # [T, C] log posteriors
    label_posteriors: Optional[np.ndarray] = None


class WordLevelCTCDecoder:
    """
    CTC prefix beam search with N-gram shallow fusion at word boundaries.

    Implements the paper's approach (Willett et al. 2023, Section F):
    the N-gram LM is integrated **during** CTC beam search, not after.

    Each beam state tracks:
      - phoneme prefix (for CTC collapsing rules)
      - current word's phoneme buffer (phones since last SIL)
      - completed word history
      - accumulated N-gram log-probability

    At each frame:
      - Blank/repeat: standard CTC rules
      - SIL emission: complete current word via lexicon lookup,
        add N-gram score, reset phoneme buffer
      - Other phoneme: extend phoneme buffer

    Beam pruning uses the **combined score**:
        score(b) = alpha * log P_enc(b) + log P_ngram(b)

    This means the N-gram influences which beams survive, allowing
    linguistically plausible paths to be kept even if their CTC
    score is slightly lower.

    Parameters
    ----------
    beam_width : int
        Number of active beams during CTC search.
    blank : int
        CTC blank index.
    sil : int
        SIL (word boundary) index.
    n_best : int
        Number of word-level hypotheses to return.
    blank_penalty : float
        Additive log-penalty for blank emissions.
    alpha : float
        CTC score weight (paper's alpha, typically 0.8).
        score = alpha * ctc + ngram
    lexicon : object or None
        PronunciationLexicon instance.
    ngram_lm : object or None
        NgramLM instance.
    max_word_candidates : int
        Maximum lexicon matches when completing a word at SIL.
    """

    LOG10 = math.log(10.0)  # ≈ 2.3026

    def __init__(
        self,
        beam_width: int = 25,
        blank: int = 0,
        sil: int = 40,
        n_best: int = 50,
        blank_penalty: float = 0.693,
        alpha: float = 0.8,
        lexicon=None,
        ngram_lm=None,
        max_word_candidates: int = 20,
        # Legacy params (ignored, kept for backward compat)
        phoneme_beam_width: int = 25,
        ngram_alpha: float = 1.25,
    ):
        self.beam_width = beam_width
        self.blank = blank
        self.sil = sil
        self.n_best = n_best
        self.blank_penalty = blank_penalty
        self.alpha = alpha
        self.lexicon = lexicon
        self.ngram_lm = ngram_lm
        self.max_word_candidates = max_word_candidates

    def decode(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None,
    ) -> List[WordCTCHypothesis]:
        """
        CTC prefix beam search with N-gram scoring at word boundaries.

        Parameters
        ----------
        log_probs : np.ndarray, shape [T, C]
            Frame-level log-softmax outputs.
        length : int or None
            Valid frame count.

        Returns
        -------
        List[WordCTCHypothesis]
            Up to *n_best* hypotheses sorted by combined score (best first).
        """
        if self.lexicon is None:
            raise ValueError(
                "WordLevelCTCDecoder requires a lexicon. "
                "Set lexicon= in the constructor."
            )

        T, C = log_probs.shape
        if length is not None:
            T = min(T, int(length))
        log_probs = log_probs[:T]

        use_ngram = self.ngram_lm is not None and self.ngram_lm.is_ready

        alpha = self.alpha

        # Apply blank penalty
        penalised = log_probs.copy()
        if self.blank_penalty != 0.0:
            penalised[:, self.blank] -= self.blank_penalty

        NEG_INF = -math.inf

        # ---- Beam state ----
        # Key: (phoneme_prefix, word_history_tuple)
        #   phoneme_prefix: tuple of phoneme IDs (CTC collapsed)
        #   word_history_tuple: tuple of completed words
        #
        # Value: [p_blank, p_non_blank, word_phones, ngram_log10]
        #   p_blank: log P(prefix, blank at t)
        #   p_non_blank: log P(prefix, non-blank at t)
        #   word_phones: list of phoneme IDs for current (incomplete) word
        #   ngram_log10: accumulated N-gram score (log10)

        # Type alias for clarity
        # state_key = (phone_prefix_tuple, word_history_tuple)
        # state_val = [p_blank, p_non_blank, word_phones_list, ngram_log10]

        beams: Dict[Tuple[Tuple[int, ...], Tuple[str, ...]], List] = {}

        # Initialize: empty prefix, no words
        empty_key = ((), ())
        beams[empty_key] = [penalised[0, self.blank], NEG_INF, [], 0.0]

        # Initialize with first-frame non-blank emissions
        for c in range(C):
            if c == self.blank:
                continue
            if c == self.sil:
                # SIL at start = still empty word buffer
                key = ((c,), ())
                beams[key] = [NEG_INF, penalised[0, c], [], 0.0]
            else:
                key = ((c,), ())
                beams[key] = [NEG_INF, penalised[0, c], [c], 0.0]

        for t in range(1, T):
            new_beams: Dict[Tuple[Tuple[int, ...], Tuple[str, ...]], List] = {}

            # Score each beam for pruning:
            # combined = alpha * ctc + ngram_ln
            def _beam_score(key):
                val = beams[key]
                ctc = _log_add(val[0], val[1])
                ng_ln = val[3] * self.LOG10  # convert log10 to ln
                return alpha * ctc + ng_ln

            scored_keys = sorted(beams.keys(), key=_beam_score, reverse=True)
            top_keys = scored_keys[: self.beam_width]

            for key in top_keys:
                phone_prefix, word_hist = key
                pb, pnb, word_phones, ng_acc = beams[key]
                p_total = _log_add(pb, pnb)

                # --- extend with blank ---
                new_pb = p_total + penalised[t, self.blank]
                if key not in new_beams:
                    new_beams[key] = [NEG_INF, NEG_INF, list(word_phones), ng_acc]
                new_beams[key][0] = _log_add(new_beams[key][0], new_pb)

                # --- extend with non-blank ---
                for c in range(C):
                    if c == self.blank:
                        continue

                    if c == self.sil:
                        # ---- SIL: word boundary ----
                        # Complete current word and start new one
                        ext_prefix = phone_prefix + (c,)

                        if word_phones:
                            # Look up word in lexicon
                            new_word, edit_dist = self._lookup_word(word_phones)
                            new_word_hist = word_hist + (new_word,)

                            # N-gram score for completed word
                            if use_ngram:
                                context = ["<s>"] + list(word_hist)
                                ng_score = self.ngram_lm.score_word(new_word, context)
                                new_ng = ng_acc + ng_score
                            else:
                                new_ng = ng_acc

                            # Penalize edit distance
                            edit_penalty = edit_dist * 0.5  # in log10 scale
                            new_ng -= edit_penalty
                        else:
                            # Empty word buffer (consecutive SILs)
                            new_word_hist = word_hist
                            new_ng = ng_acc

                        new_key = (ext_prefix, new_word_hist)

                        if phone_prefix and c == phone_prefix[-1]:
                            # SIL repeat: only blank-path can emit
                            new_pnb = pb + penalised[t, c]
                            if new_key not in new_beams:
                                new_beams[new_key] = [NEG_INF, NEG_INF, [], new_ng]
                            new_beams[new_key][1] = _log_add(
                                new_beams[new_key][1], new_pnb
                            )
                            # Also collapse repeat into current prefix
                            if key not in new_beams:
                                new_beams[key] = [
                                    NEG_INF,
                                    NEG_INF,
                                    list(word_phones),
                                    ng_acc,
                                ]
                            new_beams[key][1] = _log_add(
                                new_beams[key][1], pnb + penalised[t, c]
                            )
                        else:
                            new_pnb = p_total + penalised[t, c]
                            if new_key not in new_beams:
                                new_beams[new_key] = [NEG_INF, NEG_INF, [], new_ng]
                            new_beams[new_key][1] = _log_add(
                                new_beams[new_key][1], new_pnb
                            )

                    else:
                        # ---- Regular phoneme ----
                        ext_prefix = phone_prefix + (c,)
                        new_word_phones = list(word_phones) + [c]

                        new_key = (ext_prefix, word_hist)

                        if phone_prefix and c == phone_prefix[-1]:
                            # Same label as last: only blank-path extends
                            new_pnb = pb + penalised[t, c]
                            if new_key not in new_beams:
                                new_beams[new_key] = [
                                    NEG_INF,
                                    NEG_INF,
                                    new_word_phones,
                                    ng_acc,
                                ]
                            new_beams[new_key][1] = _log_add(
                                new_beams[new_key][1], new_pnb
                            )
                            # Collapse repeat into current prefix
                            if key not in new_beams:
                                new_beams[key] = [
                                    NEG_INF,
                                    NEG_INF,
                                    list(word_phones),
                                    ng_acc,
                                ]
                            new_beams[key][1] = _log_add(
                                new_beams[key][1], pnb + penalised[t, c]
                            )
                        else:
                            new_pnb = p_total + penalised[t, c]
                            if new_key not in new_beams:
                                new_beams[new_key] = [
                                    NEG_INF,
                                    NEG_INF,
                                    new_word_phones,
                                    ng_acc,
                                ]
                            new_beams[new_key][1] = _log_add(
                                new_beams[new_key][1], new_pnb
                            )

            beams = new_beams

        # ---- Collect N-best ----
        # Final word completion: any remaining word_phones become the last word
        results: List[WordCTCHypothesis] = []
        seen_words: set = set()

        for key in beams:
            phone_prefix, word_hist = key
            pb, pnb, word_phones, ng_acc = beams[key]
            ctc_score = _log_add(pb, pnb)

            if ctc_score == NEG_INF:
                continue

            # Complete final word if there are remaining phonemes
            final_words = list(word_hist)
            final_ng = ng_acc

            if word_phones:
                final_word, edit_dist = self._lookup_word(word_phones)
                final_words.append(final_word)
                if use_ngram:
                    context = ["<s>"] + list(word_hist)
                    ng_score = self.ngram_lm.score_word(final_word, context)
                    final_ng += ng_score
                final_ng -= edit_dist * 0.5

            combined = alpha * ctc_score + final_ng * self.LOG10

            word_key = tuple(final_words)
            if word_key in seen_words:
                continue
            seen_words.add(word_key)

            results.append(
                WordCTCHypothesis(
                    words=final_words,
                    phoneme_ids=list(phone_prefix),
                    ctc_log_prob=ctc_score,
                    ngram_log_prob=final_ng,
                    combined_score=combined,
                    frame_alignment=[],  # populated for 1-best below
                    frame_log_probs=log_probs,
                    label_posteriors=None,
                )
            )

        results.sort(key=lambda h: h.combined_score, reverse=True)
        results = results[: self.n_best]

        # Compute forced alignment for 1-best
        if results and results[0].phoneme_ids:
            alignment, _ = ctc_forced_align(
                log_probs, results[0].phoneme_ids, blank=self.blank
            )
            gamma = ctc_forward_backward(
                log_probs, results[0].phoneme_ids, blank=self.blank
            )
            results[0].frame_alignment = alignment
            results[0].label_posteriors = gamma

        return results

    def _lookup_word(self, phones: List[int]) -> Tuple[str, float]:
        """Look up best word for a phoneme chunk. Returns (word, edit_dist)."""
        candidates = self.lexicon._find_word_candidates(
            phones, max_candidates=self.max_word_candidates
        )
        if candidates:
            return candidates[0]  # best match (lowest edit distance)
        return ("<unk>", float(len(phones)))

    def decode_batch(
        self,
        log_probs_batch: np.ndarray,
        lengths: Optional[np.ndarray] = None,
    ) -> List[List[WordCTCHypothesis]]:
        """Decode a batch of utterances."""
        B = log_probs_batch.shape[0]
        results = []
        for b in range(B):
            length = int(lengths[b]) if lengths is not None else None
            results.append(self.decode(log_probs_batch[b], length))
        return results
