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
    Word-level CTC prefix beam search with N-gram shallow fusion.

    This decoder integrates lexicon lookup and N-gram scoring *during*
    beam search, not after.  When a SIL token is emitted (word boundary),
    the accumulated phoneme buffer is looked up in the lexicon, producing
    multiple word candidates.  Each candidate is scored by the N-gram LM,
    and the combined CTC + N-gram score determines which beams survive
    pruning.

    This produces **word-level diverse** N-best lists — different beams
    carry different word sequences, not just phoneme variants of the same
    words.

    Following Willett et al. (2023):
      score(b) = ctc_log_prob(b) + alpha * ngram_log10(b)

    The N-gram score uses log10 (KenLM convention); we convert to natural
    log for combination with CTC scores by multiplying by log(10).

    Parameters
    ----------
    beam_width : int
        Number of active word-level beams.
    phoneme_beam_width : int
        Number of phoneme-level prefixes to track within each word.
        Controls intra-word phoneme diversity.
    blank : int
        CTC blank index.
    sil : int
        SIL (word boundary) index.
    n_best : int
        Number of word-level hypotheses to return.
    blank_penalty : float
        Additive log-penalty for blank emissions.
    ngram_alpha : float
        Weight for N-gram log-prob during beam search.
        The N-gram score is in log10; this scales it before adding to
        the CTC log-prob (natural log).  Typical: 0.5-1.0.
    lexicon : object or None
        PronunciationLexicon instance.  Required for word lookup.
    ngram_lm : object or None
        NgramLM instance.  If None, no language model fusion.
    max_word_candidates : int
        Maximum lexicon matches per phoneme chunk.
    """

    LOG10 = math.log(10.0)  # ≈ 2.3026

    def __init__(
        self,
        beam_width: int = 25,
        phoneme_beam_width: int = 8,
        blank: int = 0,
        sil: int = 40,
        n_best: int = 50,
        blank_penalty: float = 0.693,
        ngram_alpha: float = 0.5,
        lexicon=None,
        ngram_lm=None,
        max_word_candidates: int = 8,
    ):
        self.beam_width = beam_width
        self.phoneme_beam_width = phoneme_beam_width
        self.blank = blank
        self.sil = sil
        self.n_best = n_best
        self.blank_penalty = blank_penalty
        self.ngram_alpha = ngram_alpha
        self.lexicon = lexicon
        self.ngram_lm = ngram_lm
        self.max_word_candidates = max_word_candidates

    def decode(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None,
    ) -> List[WordCTCHypothesis]:
        """
        Word-level CTC beam search on a single utterance.

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

        # Apply blank penalty
        penalised = log_probs.copy()
        if self.blank_penalty != 0.0:
            penalised[:, self.blank] -= self.blank_penalty

        NEG_INF = -math.inf
        use_ngram = (
            self.ngram_lm is not None
            and self.ngram_alpha > 0
            and self.ngram_lm.is_ready
        )

        # ---- State representation ----
        # Each beam is identified by:
        #   word_key: tuple of words decoded so far
        #   phone_buf: tuple of phoneme IDs accumulated since last SIL
        # State value:
        #   (p_blank, p_non_blank, ngram_logp, full_phone_seq)
        #
        # We use a compound key (word_key, phone_buf) to track both the
        # word-level history (for N-gram) and the current phoneme buffer
        # (for CTC prefix merging).
        #
        # Notation:
        #   p_blank, p_non_blank: CTC log-probs ending in blank / non-blank
        #   ngram_logp: accumulated N-gram log10 score for the word sequence
        #   full_phone_seq: complete phoneme sequence (for final alignment)

        # Type: Dict[(word_tuple, phone_buf_tuple), [p_blank, p_nonblank, ngram_log10, phone_list]]
        beams: Dict[
            Tuple[Tuple[str, ...], Tuple[int, ...]],
            List,  # [p_blank, p_nonblank, ngram_log10, List[int]]
        ] = {}

        # Initialise: empty word sequence, empty phone buffer
        init_key = ((), ())
        beams[init_key] = [penalised[0, self.blank], NEG_INF, 0.0, []]

        # Init with first-frame non-blank emissions
        for c in range(C):
            if c == self.blank:
                continue
            if c == self.sil:
                # SIL at frame 0 → empty word boundary (ignore)
                key = ((), ())
                if key not in beams:
                    beams[key] = [NEG_INF, NEG_INF, 0.0, []]
                beams[key][1] = _log_add(beams[key][1], penalised[0, c])
            else:
                key = ((), (c,))
                beams[key] = [NEG_INF, penalised[0, c], 0.0, [c]]

        for t in range(1, T):
            new_beams: Dict[Tuple[Tuple[str, ...], Tuple[int, ...]], List] = {}

            # Score and prune beams
            scored = {}
            for key, (pb, pnb, ng, _) in beams.items():
                ctc_total = _log_add(pb, pnb)
                # Combined score for pruning: CTC + ngram
                combined = ctc_total + self.ngram_alpha * ng * self.LOG10
                scored[key] = (ctc_total, combined)

            # Prune to beam_width by combined score
            top_keys = sorted(scored, key=lambda k: scored[k][1], reverse=True)[
                : self.beam_width
            ]

            for key in top_keys:
                word_seq, phone_buf = key
                pb, pnb, ng, full_phones = beams[key]
                p_total = _log_add(pb, pnb)

                # --- Extend with blank ---
                new_pb = p_total + penalised[t, self.blank]
                if key not in new_beams:
                    new_beams[key] = [NEG_INF, NEG_INF, ng, list(full_phones)]
                new_beams[key][0] = _log_add(new_beams[key][0], new_pb)

                # --- Extend with SIL (word boundary) ---
                sil_prob = penalised[t, self.sil]
                if sil_prob > NEG_INF + 100:  # only if SIL has non-negligible prob
                    self._handle_sil_emission(
                        new_beams,
                        word_seq,
                        phone_buf,
                        p_total + sil_prob,
                        ng,
                        full_phones,
                        use_ngram,
                        t,
                    )

                # --- Extend with non-blank, non-SIL phonemes ---
                for c in range(C):
                    if c == self.blank or c == self.sil:
                        continue
                    new_phone_buf = phone_buf + (c,)
                    new_full = full_phones + [c]
                    new_key = (word_seq, new_phone_buf)

                    if phone_buf and c == phone_buf[-1]:
                        # Same label as last in buffer — CTC repeat rule
                        # Only blank-ending path can emit this without collapse
                        new_pnb = pb + penalised[t, c]
                        if new_key not in new_beams:
                            new_beams[new_key] = [NEG_INF, NEG_INF, ng, new_full]
                        new_beams[new_key][1] = _log_add(new_beams[new_key][1], new_pnb)
                        # Also continue current buffer (collapse via non-blank)
                        stay_key = key
                        if stay_key not in new_beams:
                            new_beams[stay_key] = [
                                NEG_INF,
                                NEG_INF,
                                ng,
                                list(full_phones),
                            ]
                        new_beams[stay_key][1] = _log_add(
                            new_beams[stay_key][1], pnb + penalised[t, c]
                        )
                    else:
                        new_pnb = p_total + penalised[t, c]
                        if new_key not in new_beams:
                            new_beams[new_key] = [NEG_INF, NEG_INF, ng, new_full]
                        new_beams[new_key][1] = _log_add(new_beams[new_key][1], new_pnb)

            beams = new_beams

        # ---- Final: flush any remaining phoneme buffers ----
        final_beams = self._flush_final_buffers(beams, use_ngram)

        # ---- Collect N-best ----
        scored_final = []
        for key, (pb, pnb, ng, full_phones) in final_beams.items():
            word_seq, phone_buf = key
            ctc_total = _log_add(pb, pnb)
            combined = ctc_total + self.ngram_alpha * ng * self.LOG10
            scored_final.append(
                (word_seq, phone_buf, ctc_total, ng, combined, full_phones)
            )

        scored_final.sort(key=lambda x: x[4], reverse=True)

        # De-duplicate by word sequence (keep best combined score)
        seen_words: set = set()
        deduped = []
        for item in scored_final:
            wkey = item[0]
            if wkey not in seen_words:
                seen_words.add(wkey)
                deduped.append(item)

        results: List[WordCTCHypothesis] = []
        for rank, (word_seq, phone_buf, ctc_lp, ng_lp, comb, full_phones) in enumerate(
            deduped[: self.n_best]
        ):
            phone_ids = full_phones if full_phones else list(phone_buf)

            # Forced alignment + forward-backward only for 1-best
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
                WordCTCHypothesis(
                    words=list(word_seq),
                    phoneme_ids=phone_ids,
                    ctc_log_prob=ctc_lp,
                    ngram_log_prob=ng_lp,
                    combined_score=comb,
                    frame_alignment=alignment,
                    frame_log_probs=log_probs,
                    label_posteriors=gamma,
                )
            )

        return results

    def _handle_sil_emission(
        self,
        new_beams: Dict,
        word_seq: Tuple[str, ...],
        phone_buf: Tuple[int, ...],
        ctc_logp: float,
        ngram_logp: float,
        full_phones: List[int],
        use_ngram: bool,
        t: int,
    ):
        """
        Handle a SIL emission: flush the phoneme buffer through the lexicon.

        If the buffer is non-empty, look up word candidates and create
        new beams for each.  If empty (consecutive SILs), just continue
        the current beam.
        """
        NEG_INF = -math.inf

        if not phone_buf:
            # Empty buffer (consecutive SILs) — continue as-is
            key = (word_seq, ())
            new_phones = full_phones + [self.sil]
            if key not in new_beams:
                new_beams[key] = [NEG_INF, NEG_INF, ngram_logp, new_phones]
            # SIL acts like a blank for CTC purposes — goes to p_blank slot
            new_beams[key][0] = _log_add(new_beams[key][0], ctc_logp)
            return

        # Non-empty buffer: look up word candidates
        candidates = self.lexicon._find_word_candidates(
            list(phone_buf), max_candidates=self.max_word_candidates
        )

        if not candidates:
            # No match — keep the phoneme buffer as <unk>
            new_word_seq = word_seq + ("<unk>",)
            key = (new_word_seq, ())
            new_phones = full_phones + [self.sil]
            edit_penalty = float(len(phone_buf)) * 0.5  # penalty for unk
            # Penalise the CTC score by edit penalty (in log space)
            adjusted_ctc = ctc_logp - edit_penalty
            if use_ngram:
                ng_score = self._ngram_word_score("<unk>", word_seq)
                new_ng = ngram_logp + ng_score
            else:
                new_ng = ngram_logp
            if key not in new_beams:
                new_beams[key] = [NEG_INF, NEG_INF, new_ng, new_phones]
            new_beams[key][0] = _log_add(new_beams[key][0], adjusted_ctc)
            return

        # Expand beam for each word candidate
        for word, edit_dist in candidates:
            new_word_seq = word_seq + (word,)
            key = (new_word_seq, ())
            new_phones = full_phones + [self.sil]

            # Edit distance penalty (in natural log scale)
            # Small penalty so exact matches are preferred
            edit_penalty = edit_dist * 0.5

            adjusted_ctc = ctc_logp - edit_penalty

            if use_ngram:
                ng_score = self._ngram_word_score(word, word_seq)
                new_ng = ngram_logp + ng_score
            else:
                new_ng = ngram_logp

            if key not in new_beams:
                new_beams[key] = [NEG_INF, NEG_INF, new_ng, new_phones]
            else:
                # Keep the better ngram score if merging
                if new_ng > new_beams[key][2]:
                    new_beams[key][2] = new_ng
            new_beams[key][0] = _log_add(new_beams[key][0], adjusted_ctc)

    def _ngram_word_score(self, word: str, word_seq: Tuple[str, ...]) -> float:
        """Score a word with the N-gram LM given preceding context."""
        if self.ngram_lm is None or not self.ngram_lm.is_ready:
            return 0.0
        context = ["<s>"] + list(word_seq) if not word_seq else list(word_seq)
        return self.ngram_lm.score_word(word, context)

    def _flush_final_buffers(
        self,
        beams: Dict,
        use_ngram: bool,
    ) -> Dict:
        """
        At the end of the utterance, flush any non-empty phoneme buffers
        through the lexicon to produce final word hypotheses.
        """
        NEG_INF = -math.inf
        final = {}

        for key, (pb, pnb, ng, full_phones) in beams.items():
            word_seq, phone_buf = key

            if not phone_buf:
                # Already flushed — keep as-is
                if key not in final:
                    final[key] = [pb, pnb, ng, full_phones]
                else:
                    final[key][0] = _log_add(final[key][0], pb)
                    final[key][1] = _log_add(final[key][1], pnb)
                continue

            # Has unflushed phonemes — look up in lexicon
            ctc_total = _log_add(pb, pnb)
            candidates = self.lexicon._find_word_candidates(
                list(phone_buf), max_candidates=self.max_word_candidates
            )

            if not candidates:
                # <unk> fallback
                new_word_seq = word_seq + ("<unk>",)
                fkey = (new_word_seq, ())
                edit_penalty = float(len(phone_buf)) * 0.5
                adjusted = ctc_total - edit_penalty
                if use_ngram:
                    ng_score = self._ngram_word_score("<unk>", word_seq)
                    new_ng = ng + ng_score
                else:
                    new_ng = ng
                if fkey not in final:
                    final[fkey] = [adjusted, NEG_INF, new_ng, full_phones]
                else:
                    final[fkey][0] = _log_add(final[fkey][0], adjusted)
                continue

            for word, edit_dist in candidates:
                new_word_seq = word_seq + (word,)
                fkey = (new_word_seq, ())
                edit_penalty = edit_dist * 0.5
                adjusted = ctc_total - edit_penalty
                if use_ngram:
                    ng_score = self._ngram_word_score(word, word_seq)
                    new_ng = ng + ng_score
                else:
                    new_ng = ng
                if fkey not in final:
                    final[fkey] = [adjusted, NEG_INF, new_ng, full_phones]
                else:
                    final[fkey][0] = _log_add(final[fkey][0], adjusted)
                    if new_ng > final[fkey][2]:
                        final[fkey][2] = new_ng

        return final

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
