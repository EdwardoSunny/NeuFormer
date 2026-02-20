"""
B1 – Uncertainty Estimation
============================
Computes per-frame and per-word confidence measures from CTC posteriors.

Metrics
-------
* **Entropy**: per-frame Shannon entropy of the posterior distribution.
* **Margin**: difference between top-1 and top-2 log-probabilities.
* **Blank dominance**: fraction of frames where blank is the argmax.
* **Word-level confidence**: aggregated from frame-level stats over the
  aligned span of each word.

Usage
-----
    from neural_decoder.uncertainty import UncertaintyEstimator
    est = UncertaintyEstimator(blank_idx=0, sil_idx=40)
    frame_info = est.compute_frame_uncertainty(log_probs, length)
    word_conf = est.compute_word_confidence(frame_info, word_spans)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .phoneme_table import BLANK_IDX, SIL_IDX


@dataclass
class FrameUncertainty:
    """Per-frame uncertainty measures for one utterance."""

    entropy: np.ndarray  # [T] Shannon entropy per frame
    margin: np.ndarray  # [T] top1 - top2 log-prob
    blank_prob: np.ndarray  # [T] probability of blank
    argmax: np.ndarray  # [T] greedy argmax label
    length: int  # actual number of valid frames


@dataclass
class WordConfidence:
    """Word-level aggregated confidence for a single word span."""

    word: str
    start_frame: int
    end_frame: int
    mean_entropy: float
    min_margin: float
    blank_ratio: float
    confidence: float  # combined scalar in [0, 1]


class UncertaintyEstimator:
    """
    Computes frame-level and word-level uncertainty from CTC posteriors.

    Parameters
    ----------
    blank_idx : int
        CTC blank index (default 0).
    sil_idx : int
        SIL (word boundary) index.
    entropy_threshold : float
        Frames with entropy above this are considered "uncertain".
    margin_threshold : float
        Frames with margin below this are considered "uncertain".
    """

    def __init__(
        self,
        blank_idx: int = BLANK_IDX,
        sil_idx: int = SIL_IDX,
        entropy_threshold: float = 1.5,
        margin_threshold: float = 0.3,
    ):
        self.blank_idx = blank_idx
        self.sil_idx = sil_idx
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        # Learned logistic regression model (None = use hand-tuned fallback)
        self._confidence_model = None

    def compute_frame_uncertainty(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None,
    ) -> FrameUncertainty:
        """
        Compute per-frame uncertainty measures.

        Parameters
        ----------
        log_probs : np.ndarray, shape [T, C]
            Log-softmax posteriors.
        length : int or None
            Valid frame count.

        Returns
        -------
        FrameUncertainty
        """
        T, C = log_probs.shape
        if length is not None:
            T = min(T, int(length))
        lp = log_probs[:T]

        probs = np.exp(lp)
        # Clip for numerical safety
        probs = np.clip(probs, 1e-12, 1.0)

        # Shannon entropy per frame
        entropy = -np.sum(probs * np.log(probs), axis=-1)  # [T]

        # Margin: difference between top-1 and top-2 log-probs
        sorted_lp = np.sort(lp, axis=-1)
        margin = sorted_lp[:, -1] - sorted_lp[:, -2]  # [T]

        # Blank probability
        blank_prob = probs[:, self.blank_idx]  # [T]

        # Argmax
        argmax = np.argmax(lp, axis=-1)  # [T]

        return FrameUncertainty(
            entropy=entropy,
            margin=margin,
            blank_prob=blank_prob,
            argmax=argmax,
            length=T,
        )

    def compute_word_confidence(
        self,
        frame_info: FrameUncertainty,
        word_spans: List[Tuple[str, int, int]],
    ) -> List[WordConfidence]:
        """
        Aggregate frame-level uncertainty into word-level confidence.

        Parameters
        ----------
        frame_info : FrameUncertainty
            Frame-level stats.
        word_spans : list of (word, start_frame, end_frame)
            Word-level alignment spans.

        Returns
        -------
        list of WordConfidence
        """
        results: List[WordConfidence] = []
        for word, s, e in word_spans:
            s = max(0, s)
            e = min(frame_info.length, e)
            if e <= s:
                results.append(
                    WordConfidence(
                        word=word,
                        start_frame=s,
                        end_frame=e,
                        mean_entropy=0.0,
                        min_margin=0.0,
                        blank_ratio=0.0,
                        confidence=0.0,
                    )
                )
                continue

            span_entropy = frame_info.entropy[s:e]
            span_margin = frame_info.margin[s:e]
            span_blank = frame_info.blank_prob[s:e]
            span_argmax = frame_info.argmax[s:e]

            mean_ent = float(np.mean(span_entropy))
            min_mar = float(np.min(span_margin))
            blank_ratio = float(np.mean(span_argmax == self.blank_idx))

            # Combined confidence: high margin + low entropy + low blank → high confidence
            # Sigmoid-like mapping
            conf = self._compute_confidence(mean_ent, min_mar, blank_ratio)

            results.append(
                WordConfidence(
                    word=word,
                    start_frame=s,
                    end_frame=e,
                    mean_entropy=mean_ent,
                    min_margin=min_mar,
                    blank_ratio=blank_ratio,
                    confidence=conf,
                )
            )

        return results

    def _compute_confidence(
        self,
        mean_entropy: float,
        min_margin: float,
        blank_ratio: float,
        duration: Optional[int] = None,
    ) -> float:
        """
        Combine uncertainty features into a single confidence score in [0, 1].

        Uses a **learned logistic regression** if one has been fitted
        (via ``fit_confidence_model``), otherwise falls back to hand-tuned
        weights as a reasonable default.

        Features: [margin, entropy, blank_ratio, log(duration+1)]
        """
        if self._confidence_model is not None:
            return self._predict_confidence_learned(
                mean_entropy, min_margin, blank_ratio, duration
            )
        return self._compute_confidence_handtuned(mean_entropy, min_margin, blank_ratio)

    def _compute_confidence_handtuned(
        self, mean_entropy: float, min_margin: float, blank_ratio: float
    ) -> float:
        """
        Hand-tuned fallback confidence function.

        Uses a simple weighted logistic combination:
          logit = w_margin * margin - w_entropy * entropy - w_blank * blank_ratio
          confidence = sigmoid(logit)

        These weights are reasonable defaults but should be replaced
        with a learned model when development data is available (see
        ``fit_confidence_model``).
        """
        w_margin = 3.0
        w_entropy = 1.5
        w_blank = 2.0
        bias = 0.0
        logit = (
            w_margin * min_margin
            - w_entropy * mean_entropy
            - w_blank * blank_ratio
            + bias
        )
        return 1.0 / (1.0 + np.exp(-logit))

    def _predict_confidence_learned(
        self,
        mean_entropy: float,
        min_margin: float,
        blank_ratio: float,
        duration: Optional[int] = None,
    ) -> float:
        """Use the fitted logistic regression model."""
        dur_feat = np.log(max(duration or 1, 1) + 1)
        features = np.array(
            [[min_margin, mean_entropy, blank_ratio, dur_feat]],
            dtype=np.float64,
        )
        prob = self._confidence_model.predict_proba(features)[0, 1]
        return float(prob)

    def fit_confidence_model(
        self,
        word_features: List[Tuple[float, float, float, int]],
        word_correct: List[bool],
    ) -> dict:
        """
        Learn the confidence → correctness mapping from development data.

        Trains a logistic regression predicting whether a word matches
        the reference, using uncertainty features:
          [min_margin, mean_entropy, blank_ratio, log(duration+1)]

        Parameters
        ----------
        word_features : list of (min_margin, mean_entropy, blank_ratio, duration)
            One tuple per word from the development set.
        word_correct : list of bool
            Whether each word matched the reference.

        Returns
        -------
        dict with keys:
            'accuracy': float — training accuracy
            'coefficients': np.ndarray — learned weights
            'intercept': float — learned bias
            'n_samples': int
        """
        from sklearn.linear_model import LogisticRegression

        X = np.array(
            [[m, e, b, np.log(max(d, 1) + 1)] for m, e, b, d in word_features],
            dtype=np.float64,
        )
        y = np.array(word_correct, dtype=np.int32)

        if len(np.unique(y)) < 2:
            # Can't fit with only one class — keep hand-tuned
            return {
                "accuracy": 0.0,
                "coefficients": None,
                "intercept": 0.0,
                "n_samples": len(y),
            }

        model = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
        model.fit(X, y)
        self._confidence_model = model

        accuracy = float(model.score(X, y))
        return {
            "accuracy": accuracy,
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "n_samples": len(y),
        }

    def classify_word_confidence(
        self,
        word_confs: List[WordConfidence],
        high_threshold: float = 0.7,
        low_threshold: float = 0.4,
    ) -> List[Tuple[WordConfidence, str]]:
        """
        Classify each word as "high", "medium", or "low" confidence.

        Returns (WordConfidence, label) pairs.
        """
        results = []
        for wc in word_confs:
            if wc.confidence >= high_threshold:
                label = "high"
            elif wc.confidence >= low_threshold:
                label = "medium"
            else:
                label = "low"
            results.append((wc, label))
        return results

    def get_word_spans_from_forced_alignment(
        self,
        frame_alignment: List[int],
        phoneme_ids: List[int],
        words: List[str],
        phoneme_to_word_map: Optional[List[int]] = None,
    ) -> List[Tuple[str, int, int]]:
        """
        Extract word-level frame spans from a CTC forced alignment.

        Unlike the old argmax-based method, this uses a proper Viterbi
        forced alignment (constrained to the hypothesis sequence), which
        guarantees the alignment is consistent with the hypothesis.

        Parameters
        ----------
        frame_alignment : list of int
            Per-frame label from forced alignment (blanks + hypothesis labels).
        phoneme_ids : list of int
            Decoded phoneme sequence (no blanks, no repeats).
        words : list of str
            Word sequence corresponding to the phoneme sequence.
        phoneme_to_word_map : list of int, optional
            For each phoneme in phoneme_ids, which word index it belongs to.
            If None, derived from splitting phoneme_ids on SIL.

        Returns
        -------
        list of (word, start_frame, end_frame)
        """
        if not words or not phoneme_ids:
            return []

        n_words = len(words)

        # If no explicit phone-to-word map, derive from SIL boundaries
        if phoneme_to_word_map is None:
            phoneme_to_word_map = self._derive_phone_to_word_map(phoneme_ids, n_words)

        # Extract per-phoneme frame spans from the forced alignment
        # The forced alignment assigns exactly one label per frame, and
        # the labels are from the hypothesis (or blank).
        phone_frame_starts: List[int] = []
        phone_frame_ends: List[int] = []

        phone_idx = 0
        in_phone = False
        start = 0
        T = len(frame_alignment)

        for f in range(T):
            label = frame_alignment[f]
            if phone_idx >= len(phoneme_ids):
                break

            if label == phoneme_ids[phone_idx]:
                if not in_phone:
                    start = f
                    in_phone = True
            else:
                if in_phone:
                    phone_frame_starts.append(start)
                    phone_frame_ends.append(f)
                    phone_idx += 1
                    in_phone = False
                    # Check if this frame starts the next phoneme
                    if phone_idx < len(phoneme_ids) and label == phoneme_ids[phone_idx]:
                        start = f
                        in_phone = True

        # Handle last phoneme still active
        if in_phone and phone_idx < len(phoneme_ids):
            phone_frame_starts.append(start)
            phone_frame_ends.append(T)
            phone_idx += 1

        # Fill remaining (shouldn't happen with proper forced alignment)
        while len(phone_frame_starts) < len(phoneme_ids):
            last_end = phone_frame_ends[-1] if phone_frame_ends else 0
            phone_frame_starts.append(last_end)
            phone_frame_ends.append(min(last_end + 1, T))

        # Aggregate phoneme spans into word spans
        word_spans: List[Tuple[str, int, int]] = []
        for w_idx in range(n_words):
            phone_indices = [
                i for i, wm in enumerate(phoneme_to_word_map) if wm == w_idx
            ]
            if not phone_indices:
                if word_spans:
                    last_end = word_spans[-1][2]
                else:
                    last_end = 0
                word_spans.append((words[w_idx], last_end, min(last_end + 1, T)))
                continue

            valid_indices = [i for i in phone_indices if i < len(phone_frame_starts)]
            if not valid_indices:
                if word_spans:
                    last_end = word_spans[-1][2]
                else:
                    last_end = 0
                word_spans.append((words[w_idx], last_end, min(last_end + 1, T)))
                continue

            w_start = phone_frame_starts[valid_indices[0]]
            w_end = phone_frame_ends[valid_indices[-1]]
            word_spans.append((words[w_idx], w_start, w_end))

        return word_spans

    def _derive_phone_to_word_map(
        self, phoneme_ids: List[int], n_words: int
    ) -> List[int]:
        """
        Derive a phoneme-to-word mapping by splitting on SIL tokens.

        Each phoneme between SIL boundaries is assigned to the next word.
        """
        mapping: List[int] = []
        word_idx = 0
        for pid in phoneme_ids:
            if pid == self.sil_idx:
                word_idx = min(word_idx + 1, n_words - 1)
            else:
                mapping.append(min(word_idx, n_words - 1))
        return mapping

    def compute_word_confidence_from_posteriors(
        self,
        frame_info: FrameUncertainty,
        label_posteriors: np.ndarray,
        phoneme_ids: List[int],
        words: List[str],
        phoneme_to_word_map: Optional[List[int]] = None,
    ) -> List[WordConfidence]:
        """
        Compute word-level confidence using forward-backward posteriors.

        This is more principled than the frame-level entropy/margin
        approach: for each word, we sum the posterior mass of its
        constituent labels across their aligned frames.

        Parameters
        ----------
        frame_info : FrameUncertainty
            Standard frame-level stats.
        label_posteriors : np.ndarray, shape [T, S]
            Forward-backward posteriors (gamma) from ctc_forward_backward.
        phoneme_ids : list of int
            Hypothesis phoneme sequence.
        words : list of str
            Word sequence.
        phoneme_to_word_map : list of int, optional

        Returns
        -------
        list of WordConfidence
        """
        if not words or not phoneme_ids:
            return []

        n_words = len(words)
        if phoneme_to_word_map is None:
            phoneme_to_word_map = self._derive_phone_to_word_map(phoneme_ids, n_words)

        T, S = label_posteriors.shape
        # Map phoneme index to expanded-label index: phoneme i -> state 2*i+1
        # (expanded = [blank, p0, blank, p1, ..., blank])

        results: List[WordConfidence] = []
        for w_idx in range(n_words):
            phone_indices = [
                i for i, wm in enumerate(phoneme_to_word_map) if wm == w_idx
            ]
            if not phone_indices:
                results.append(
                    WordConfidence(
                        word=words[w_idx],
                        start_frame=0,
                        end_frame=0,
                        mean_entropy=0.0,
                        min_margin=0.0,
                        blank_ratio=0.0,
                        confidence=0.0,
                    )
                )
                continue

            # Sum posterior mass for each phoneme's label state
            total_posterior = 0.0
            n_phones_in_word = 0
            start_frame = T
            end_frame = 0

            for pi in phone_indices:
                expanded_idx = 2 * pi + 1  # label state in expanded sequence
                if expanded_idx >= S:
                    continue
                # Sum posterior across all frames for this label state
                state_posterior = float(np.sum(label_posteriors[:, expanded_idx]))
                total_posterior += state_posterior
                n_phones_in_word += 1

                # Find frame range where this label has non-trivial posterior
                nonzero = np.where(label_posteriors[:, expanded_idx] > 0.01)[0]
                if len(nonzero) > 0:
                    start_frame = min(start_frame, int(nonzero[0]))
                    end_frame = max(end_frame, int(nonzero[-1]) + 1)

            if start_frame >= end_frame:
                start_frame = 0
                end_frame = min(1, T)

            # Also compute traditional frame-level features over the span
            s, e = max(0, start_frame), min(T, end_frame)
            if e > s and s < frame_info.length:
                e = min(e, frame_info.length)
                mean_ent = float(np.mean(frame_info.entropy[s:e]))
                min_mar = float(np.min(frame_info.margin[s:e]))
                blank_ratio = float(np.mean(frame_info.argmax[s:e] == self.blank_idx))
            else:
                mean_ent = 0.0
                min_mar = 0.0
                blank_ratio = 0.0

            # Confidence: blend posterior-based and feature-based
            # Mean posterior per phoneme in word (higher = more confident)
            mean_post = total_posterior / max(n_phones_in_word, 1)
            # Normalise: posterior is sum over frames, so normalise by T
            post_confidence = min(1.0, mean_post / max(T * 0.05, 1))
            feature_confidence = self._compute_confidence(
                mean_ent, min_mar, blank_ratio
            )
            # Blend: weight posterior-based confidence more heavily
            confidence = 0.6 * post_confidence + 0.4 * float(feature_confidence)

            results.append(
                WordConfidence(
                    word=words[w_idx],
                    start_frame=start_frame,
                    end_frame=end_frame,
                    mean_entropy=mean_ent,
                    min_margin=min_mar,
                    blank_ratio=blank_ratio,
                    confidence=confidence,
                )
            )

        return results

    # Legacy method kept for backwards compatibility
    def get_word_spans_from_alignment(
        self,
        frame_alignment: List[int],
        phoneme_ids: List[int],
        words: List[str],
        phoneme_to_word_map: List[int],
    ) -> List[Tuple[str, int, int]]:
        """
        Legacy method: estimate word spans from alignment.
        Delegates to get_word_spans_from_forced_alignment.
        """
        return self.get_word_spans_from_forced_alignment(
            frame_alignment, phoneme_ids, words, phoneme_to_word_map
        )
