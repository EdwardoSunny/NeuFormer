"""
Full Posterior-Constrained Decoding Pipeline
=============================================
Orchestrates the entire decoding pipeline following Willett et al. (2023).

Two primary modes:

**Online mode** (default, real-time capable):
  Word-level CTC beam search with N-gram shallow fusion.
  score(b) = alpha * log P_enc(b) + log P_ngram(b)
  Hyperparams: 3-gram, beam=18, alpha=0.8, blank_penalty=log(2)

**Offline mode** (higher accuracy, higher compute):
  Step A: 5-gram beam search with blank_penalty=log(7)
  Step B: Rescore beams with unpruned 5-gram
  Step C: Top-K=100 LLM rescoring
  score(b) = alpha * log P_enc(b) + beta * log P_ngram(b)
           + (1-beta) * log P_LLM(b)
  Hyperparams: alpha=0.8, beta=0.5

Legacy modes (backward-compatible):
  neural_only, unconstrained_rescore, constrained_rescore,
  slot_filling, student

Usage
-----
    from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig
    config = PipelineConfig(decode_mode="online")
    pipe = DecodePipeline(config)
    pipe.setup(training_sentences)
    result = pipe.decode_utterance(log_probs, length)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ctc_beam_decoder import (
    CTCBeamDecoder,
    CTCHypothesis,
    WordLevelCTCDecoder,
    WordCTCHypothesis,
)
from .lexicon import PronunciationLexicon
from .uncertainty import UncertaintyEstimator, FrameUncertainty, WordConfidence
from .constrained_decode import (
    ConstrainedHypothesisBuilder,
    ConstrainedTemplate,
    SlotFillingDecoder,
)
from .llm_scorer import LLMScorer
from .ngram_lm import NgramLM
from .distillation import (
    CandidateFeatures,
    DistillationDataCollector,
    DistillationDataset,
    DistillationSample,
    StudentReranker,
)
from .phoneme_table import BLANK_IDX, SIL_IDX


LOG10 = math.log(10.0)  # ≈ 2.3026


@dataclass
class PipelineConfig:
    """Configuration for the full decoding pipeline."""

    # ---- Mode ----
    # "online"  = paper's default: word-beam + N-gram, no LLM
    # "offline" = paper's offline: word-beam + N-gram + LLM rescore
    # Legacy modes: "neural_only", "unconstrained_rescore",
    #   "constrained_rescore", "slot_filling", "student"
    decode_mode: str = "online"

    # ---- CTC beam search (phoneme-level, used by legacy modes) ----
    beam_width: int = 18
    n_best: int = 100
    blank_penalty: float = 0.693  # log(2), online default

    # ---- Lexicon ----
    max_edit_dist: int = 1
    lexicon_beam_width: int = 5

    # ---- N-gram LM ----
    ngram_order: int = 3  # 3 for online, 5 for offline
    use_ngram: bool = True
    # alpha: weight for CTC log-prob relative to N-gram
    # In the paper: score = alpha * log P_enc + log P_ngram
    # We implement as: combined = ctc_logp + (1/alpha) * ngram_logp_ln
    # Or equivalently, we store alpha and ngram_alpha separately.
    alpha: float = 0.8  # CTC weight (paper's alpha)

    # ---- Word-level beam search ----
    use_word_beam: bool = True
    word_beam_width: int = 18  # paper: 18
    word_max_candidates: int = 8

    # ---- Offline-specific ----
    offline_ngram_order: int = 5  # 5-gram for offline first pass
    offline_blank_penalty: float = 1.9459  # log(7) ≈ 1.9459
    offline_top_k: int = 100  # top-K for LLM rescoring
    beta: float = 0.5  # N-gram vs LLM mixture weight

    # ---- LLM (only used in offline mode) ----
    llm_model_name: str = "meta-llama/Meta-Llama-3-8B"
    llm_device: str = "cpu"
    llm_load_in_8bit: bool = True  # paper uses 8-bit OPT

    # ---- Uncertainty (kept for backward compat) ----
    entropy_threshold: float = 1.5
    margin_threshold: float = 0.3

    # ---- Constrained hypothesis (legacy modes) ----
    high_confidence_threshold: float = 0.85
    max_candidates_per_slot: int = 20

    # ---- Legacy LLM rescoring params ----
    lambda_neural: float = 0.5
    lambda_lm: float = 1.0
    lambda_ngram: float = 1.0
    gamma_constraint: float = 1.0
    length_penalty_beta: float = 0.1
    slot_filling_beam: int = 10
    normalize_scores: bool = True
    two_pass_top_k: int = 10
    max_word_candidates: int = 100

    # ---- Legacy N-gram params ----
    ngram_weight: float = 0.5  # weight during lexicon beam search (legacy)
    phoneme_beam_width: int = 8
    word_ngram_alpha: float = 0.5

    # ---- Student (legacy) ----
    use_student: bool = False
    student_model_path: Optional[str] = None


@dataclass
class DecodeResult:
    """Result from decoding one utterance."""

    # Phoneme level
    phoneme_hypotheses: List[CTCHypothesis] = field(default_factory=list)
    best_phoneme_ids: List[int] = field(default_factory=list)
    phoneme_log_prob: float = 0.0

    # Word level (from word-beam search or lexicon lookup)
    word_hypotheses: List[Tuple[List[str], float]] = field(default_factory=list)
    best_words: List[str] = field(default_factory=list)

    # Word-level beam hypotheses (when using WordLevelCTCDecoder)
    word_beam_hypotheses: List[WordCTCHypothesis] = field(default_factory=list)

    # Uncertainty
    frame_uncertainty: Optional[FrameUncertainty] = None
    word_confidences: List[WordConfidence] = field(default_factory=list)

    # Constrained (legacy)
    template: Optional[ConstrainedTemplate] = None

    # Final output
    final_words: List[str] = field(default_factory=list)
    final_sentence: str = ""
    final_score: float = 0.0

    # Timing
    ctc_decode_time: float = 0.0
    lexicon_time: float = 0.0
    uncertainty_time: float = 0.0
    llm_time: float = 0.0
    total_time: float = 0.0


class DecodePipeline:
    """
    Full decoding pipeline following Willett et al. (2023).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Phoneme-level CTC decoder (for legacy modes or offline fallback)
        self.ctc_decoder = CTCBeamDecoder(
            beam_width=config.beam_width,
            blank=BLANK_IDX,
            n_best=config.n_best,
            blank_penalty=config.blank_penalty,
        )

        # Lexicon
        self.lexicon = PronunciationLexicon(
            max_edit_dist=config.max_edit_dist,
            sil_idx=SIL_IDX,
        )

        # Uncertainty estimator (for legacy modes)
        self.uncertainty = UncertaintyEstimator(
            blank_idx=BLANK_IDX,
            sil_idx=SIL_IDX,
            entropy_threshold=config.entropy_threshold,
            margin_threshold=config.margin_threshold,
        )

        # Constrained hypothesis builder (for legacy modes)
        self.hyp_builder = ConstrainedHypothesisBuilder(
            high_confidence_threshold=config.high_confidence_threshold,
            max_candidates_per_slot=config.max_candidates_per_slot,
        )

        # N-gram LMs
        self._ngram_lm: Optional[NgramLM] = None  # online (3-gram)
        self._ngram_lm_offline: Optional[NgramLM] = None  # offline (5-gram)

        # Word-level CTC decoders
        self._word_decoder: Optional[WordLevelCTCDecoder] = None  # online
        self._word_decoder_offline: Optional[WordLevelCTCDecoder] = None  # offline

        # LLM (lazy-loaded, only for offline mode)
        self._llm_scorer: Optional[LLMScorer] = None

        # Student (legacy)
        self._student: Optional[StudentReranker] = None

    def setup(self, training_sentences: List[str]):
        """Build the lexicon and N-gram LMs from training data."""
        self.lexicon.build_from_sentences(training_sentences)

        mode = self.config.decode_mode

        if mode in ("online", "offline") or self.config.use_ngram:
            # Train online N-gram (3-gram by default)
            self._ngram_lm = NgramLM(order=self.config.ngram_order)
            self._ngram_lm.train(training_sentences)

            # Build online word-level decoder
            # Paper score: score(b) = alpha * log P_enc(b) + log P_ngram(b)
            # We implement this as: combined = ctc_logp + (1/alpha) * ngram_logp_ln
            # But it's cleaner to just set ngram_alpha = 1/alpha so that:
            #   combined = ctc_logp + ngram_alpha * ngram_logp_in_ln
            # Since N-gram returns log10, and CTC is in ln:
            #   ngram_alpha converts: ngram_alpha * log10(P) * ln(10) = log(P)/alpha
            # Actually, the paper formula is:
            #   score = alpha * log_enc + log_ngram
            # Both in the same log base. Let's use natural log throughout.
            # ngram_alpha = 1.0 / alpha means:
            #   alpha * ctc + ngram = alpha * (ctc + (1/alpha) * ngram)
            # We can just store the raw score as alpha*ctc + ngram.
            # In WordLevelCTCDecoder, combined = ctc + ngram_alpha * ng * LOG10
            # So we want: alpha * ctc + ngram_ln
            # = alpha * (ctc + (1/alpha) * ngram_ln)
            # Since the decoder already multiplies ctc by 1 (not alpha), we need
            # to weight ngram relative to ctc. If we set ngram_alpha = 1/alpha:
            #   combined = ctc + (1/alpha) * ngram_ln
            # Then multiply final score by alpha to get: alpha*ctc + ngram_ln ✓
            #
            # Simpler: just set ngram_alpha = 1.0/alpha. The relative ranking
            # is what matters for beam pruning, and the final score from the
            # decoder's combined_score already has the right ordering.

            ngram_alpha_online = (
                1.0 / self.config.alpha if self.config.alpha > 0 else 1.0
            )

            self._word_decoder = WordLevelCTCDecoder(
                beam_width=self.config.word_beam_width,
                phoneme_beam_width=self.config.phoneme_beam_width,
                blank=BLANK_IDX,
                sil=SIL_IDX,
                n_best=self.config.n_best,
                blank_penalty=self.config.blank_penalty,
                ngram_alpha=ngram_alpha_online,
                lexicon=self.lexicon,
                ngram_lm=self._ngram_lm,
                max_word_candidates=self.config.word_max_candidates,
            )

        if mode == "offline":
            # Train offline N-gram (5-gram)
            self._ngram_lm_offline = NgramLM(order=self.config.offline_ngram_order)
            self._ngram_lm_offline.train(training_sentences)

            ngram_alpha_offline = (
                1.0 / self.config.alpha if self.config.alpha > 0 else 1.0
            )

            self._word_decoder_offline = WordLevelCTCDecoder(
                beam_width=self.config.word_beam_width,
                phoneme_beam_width=self.config.phoneme_beam_width,
                blank=BLANK_IDX,
                sil=SIL_IDX,
                n_best=self.config.n_best,
                blank_penalty=self.config.offline_blank_penalty,
                ngram_alpha=ngram_alpha_offline,
                lexicon=self.lexicon,
                ngram_lm=self._ngram_lm_offline,
                max_word_candidates=self.config.word_max_candidates,
            )

    def _get_llm(self) -> LLMScorer:
        if self._llm_scorer is None:
            self._llm_scorer = LLMScorer(
                model_name=self.config.llm_model_name,
                device=self.config.llm_device,
                load_in_8bit=self.config.llm_load_in_8bit,
            )
        return self._llm_scorer

    def _get_student(self) -> StudentReranker:
        if self._student is None:
            import torch

            self._student = StudentReranker(feature_dim=CandidateFeatures.feature_dim())
            if self.config.student_model_path:
                state = torch.load(self.config.student_model_path, map_location="cpu")
                self._student.load_state_dict(state)
            self._student.eval()
        return self._student

    # ------------------------------------------------------------------
    # Main decode entry point
    # ------------------------------------------------------------------

    def decode_utterance(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None,
    ) -> DecodeResult:
        """
        Full decoding pipeline for one utterance.

        Parameters
        ----------
        log_probs : np.ndarray, shape [T, C]
            Frame-level log-softmax from the Conformer.
        length : int or None
            Valid frame count.

        Returns
        -------
        DecodeResult
        """
        mode = self.config.decode_mode

        if mode == "online":
            return self._decode_online(log_probs, length)
        elif mode == "offline":
            return self._decode_offline(log_probs, length)
        else:
            # Legacy modes
            return self._decode_legacy(log_probs, length)

    # ------------------------------------------------------------------
    # Online mode: word-beam + N-gram, no LLM
    # ------------------------------------------------------------------

    def _decode_online(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None,
    ) -> DecodeResult:
        """
        Paper's online setup:
          score(b) = alpha * log P_enc(b) + log P_ngram(b)
          3-gram, beam=18, alpha=0.8, blank_penalty=log(2)
        """
        result = DecodeResult()
        t0 = time.time()

        if self._word_decoder is None:
            raise RuntimeError(
                "Word-level decoder not initialized. Call setup() first."
            )

        # Word-level CTC beam search with N-gram shallow fusion
        t_ctc = time.time()
        word_hyps = self._word_decoder.decode(log_probs, length)
        result.word_beam_hypotheses = word_hyps
        result.ctc_decode_time = time.time() - t_ctc

        if word_hyps:
            best = word_hyps[0]
            result.best_words = best.words
            result.best_phoneme_ids = best.phoneme_ids
            result.phoneme_log_prob = best.ctc_log_prob
            result.final_words = best.words
            result.final_score = best.combined_score

            # Also populate word_hypotheses for oracle WER computation
            # and compatibility with evaluation code
            result.word_hypotheses = [(wh.words, wh.combined_score) for wh in word_hyps]

            # Phoneme hypothesis for the 1-best (for uncertainty estimation)
            result.phoneme_hypotheses = [
                CTCHypothesis(
                    phoneme_ids=best.phoneme_ids,
                    log_prob=best.ctc_log_prob,
                    frame_alignment=best.frame_alignment,
                    frame_log_probs=best.frame_log_probs,
                    label_posteriors=best.label_posteriors,
                )
            ]

        # Uncertainty estimation (useful for diagnostics even without LLM)
        t_unc = time.time()
        frame_info = self.uncertainty.compute_frame_uncertainty(log_probs, length)
        result.frame_uncertainty = frame_info

        if result.best_words and result.phoneme_hypotheses:
            best_hyp = result.phoneme_hypotheses[0]
            if best_hyp.label_posteriors is not None:
                word_confs = self.uncertainty.compute_word_confidence_from_posteriors(
                    frame_info,
                    best_hyp.label_posteriors,
                    result.best_phoneme_ids,
                    result.best_words,
                )
            else:
                word_spans = self._estimate_word_spans(
                    frame_info, result.best_phoneme_ids, result.best_words
                )
                word_confs = self.uncertainty.compute_word_confidence(
                    frame_info, word_spans
                )
            result.word_confidences = word_confs
        result.uncertainty_time = time.time() - t_unc

        result.final_sentence = " ".join(result.final_words)
        result.total_time = time.time() - t0
        return result

    # ------------------------------------------------------------------
    # Offline mode: 5-gram beam + unpruned rescore + LLM rescore
    # ------------------------------------------------------------------

    def _decode_offline(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None,
    ) -> DecodeResult:
        """
        Paper's offline setup:
          Step A: 5-gram beam search, blank_penalty=log(7)
          Step B: Rescore with unpruned 5-gram (we use the same 5-gram
                  since we don't have a separate pruned model)
          Step C: Top-K=100 LLM rescoring
          score(b) = alpha * log P_enc(b)
                   + beta * log P_ngram(b)
                   + (1-beta) * log P_LLM(b)
          alpha=0.8, beta=0.5
        """
        result = DecodeResult()
        t0 = time.time()

        decoder = self._word_decoder_offline or self._word_decoder
        if decoder is None:
            raise RuntimeError(
                "Word-level decoder not initialized. Call setup() first."
            )

        # Step A: Word-level CTC beam search with 5-gram
        t_ctc = time.time()
        word_hyps = decoder.decode(log_probs, length)
        result.word_beam_hypotheses = word_hyps
        result.ctc_decode_time = time.time() - t_ctc

        if not word_hyps:
            result.total_time = time.time() - t0
            return result

        best = word_hyps[0]
        result.best_words = best.words
        result.best_phoneme_ids = best.phoneme_ids
        result.phoneme_log_prob = best.ctc_log_prob
        result.phoneme_hypotheses = [
            CTCHypothesis(
                phoneme_ids=best.phoneme_ids,
                log_prob=best.ctc_log_prob,
                frame_alignment=best.frame_alignment,
                frame_log_probs=best.frame_log_probs,
                label_posteriors=best.label_posteriors,
            )
        ]

        # Step B: Rescore with unpruned 5-gram
        # The word-beam search already used the 5-gram during search.
        # For the "unpruned rescore" step, we re-score the full sentences
        # with the 5-gram to get a potentially better score (the beam
        # search only conditions on a limited context window).
        ngram_lm = self._ngram_lm_offline or self._ngram_lm

        # Collect top-K candidates for LLM rescoring
        top_k = min(len(word_hyps), self.config.offline_top_k)
        candidates = word_hyps[:top_k]

        # Populate word_hypotheses for oracle WER
        result.word_hypotheses = [(wh.words, wh.combined_score) for wh in word_hyps]

        # Step C: LLM rescoring
        t_llm = time.time()
        llm = self._get_llm()

        alpha = self.config.alpha
        beta = self.config.beta

        sentences = [" ".join(wh.words) for wh in candidates]

        # LLM scores (expensive)
        llm_scores = llm.score_batch(sentences)

        # Full-sentence N-gram rescoring (Step B: "unpruned" rescore)
        if ngram_lm is not None and ngram_lm.is_ready:
            ngram_scores_raw = [ngram_lm.score_sentence_raw(s) for s in sentences]
        else:
            ngram_scores_raw = [wh.ngram_log_prob for wh in candidates]

        # Combine scores using the paper's formula:
        #   score(b) = alpha * log P_enc(b)
        #            + beta * log P_ngram(b)
        #            + (1-beta) * log P_LLM(b)
        #
        # P_enc is in natural log (CTC output).
        # P_ngram is in log10 (KenLM/python N-gram convention).
        # P_LLM is in natural log (from LLMScorer).
        #
        # Convert N-gram to natural log for consistency:
        #   ln(P_ngram) = log10(P_ngram) * ln(10)

        rescored = []
        for i, wh in enumerate(candidates):
            n_words = max(len(wh.words), 1)

            # Normalize all scores per-word for consistent scale:
            #
            # CTC: total log-prob (natural log) → per-word
            ctc_per_word = wh.ctc_log_prob / n_words
            #
            # N-gram: total log10 score → per-word, converted to ln
            ngram_per_word_ln = (ngram_scores_raw[i] / n_words) * LOG10
            #
            # LLM: score_batch() returns per-BPE-token normalized ln score.
            # This is already on a per-unit scale, so use directly.
            # (n_bpe_tokens ≈ n_words for most English text, so this is
            #  close to per-word. Trying to undo with n_words would be
            #  wrong since n_bpe_tokens ≠ n_words.)
            llm_per_unit = llm_scores[i]

            # Paper's formula (all per-word scale):
            #   score = alpha * log P_enc + beta * log P_ngram
            #         + (1-beta) * log P_LLM
            combined = (
                alpha * ctc_per_word
                + beta * ngram_per_word_ln
                + (1 - beta) * llm_per_unit
            )

            rescored.append((wh.words, combined, i))

        rescored.sort(key=lambda x: x[1], reverse=True)
        result.llm_time = time.time() - t_llm

        if rescored:
            result.final_words = rescored[0][0]
            result.final_score = rescored[0][1]

        # Uncertainty estimation
        t_unc = time.time()
        frame_info = self.uncertainty.compute_frame_uncertainty(log_probs, length)
        result.frame_uncertainty = frame_info
        if result.best_words and result.phoneme_hypotheses:
            best_hyp_phon = result.phoneme_hypotheses[0]
            if best_hyp_phon.label_posteriors is not None:
                word_confs = self.uncertainty.compute_word_confidence_from_posteriors(
                    frame_info,
                    best_hyp_phon.label_posteriors,
                    result.best_phoneme_ids,
                    result.best_words,
                )
            else:
                word_spans = self._estimate_word_spans(
                    frame_info, result.best_phoneme_ids, result.best_words
                )
                word_confs = self.uncertainty.compute_word_confidence(
                    frame_info, word_spans
                )
            result.word_confidences = word_confs
        result.uncertainty_time = time.time() - t_unc

        result.final_sentence = " ".join(result.final_words)
        result.total_time = time.time() - t0
        return result

    # ------------------------------------------------------------------
    # Legacy modes (backward-compatible)
    # ------------------------------------------------------------------

    def _decode_legacy(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None,
    ) -> DecodeResult:
        """Legacy decode modes: neural_only, unconstrained_rescore, etc."""
        result = DecodeResult()
        t0 = time.time()
        mode = self.config.decode_mode

        # Check if we should use word-beam search even for legacy modes
        use_word_beam = (
            self._word_decoder is not None
            and self.config.use_word_beam
            and mode in ("neural_only",)
        )

        if use_word_beam:
            # Use word-level beam search (same as online but with legacy output)
            t_ctc = time.time()
            word_hyps = self._word_decoder.decode(log_probs, length)
            result.word_beam_hypotheses = word_hyps
            result.ctc_decode_time = time.time() - t_ctc

            if word_hyps:
                best = word_hyps[0]
                result.best_words = best.words
                result.best_phoneme_ids = best.phoneme_ids
                result.phoneme_log_prob = best.ctc_log_prob
                result.word_hypotheses = [
                    (wh.words, wh.combined_score) for wh in word_hyps
                ]
                result.phoneme_hypotheses = [
                    CTCHypothesis(
                        phoneme_ids=best.phoneme_ids,
                        log_prob=best.ctc_log_prob,
                        frame_alignment=best.frame_alignment,
                        frame_log_probs=best.frame_log_probs,
                        label_posteriors=best.label_posteriors,
                    )
                ]
        else:
            # Phoneme-level CTC beam search + lexicon (old path)
            t_ctc = time.time()
            hyps = self.ctc_decoder.decode(log_probs, length)
            result.phoneme_hypotheses = hyps
            if hyps:
                result.best_phoneme_ids = hyps[0].phoneme_ids
                result.phoneme_log_prob = hyps[0].log_prob
            result.ctc_decode_time = time.time() - t_ctc

            # Phoneme → Word
            t_lex = time.time()
            word_hyps_list: List[Tuple[List[str], float]] = []
            T_frames = (
                log_probs.shape[0]
                if length is None
                else min(log_probs.shape[0], length)
            )
            for hyp in hyps:
                w_results = self.lexicon.phonemes_to_words(
                    hyp.phoneme_ids,
                    beam_width=self.config.lexicon_beam_width,
                    ngram_lm=self._ngram_lm,
                    ngram_weight=self.config.ngram_weight,
                )
                norm_ctc = hyp.log_prob / max(T_frames, 1)
                for words, edit_cost in w_results:
                    combined = norm_ctc - edit_cost
                    word_hyps_list.append((words, combined))

            # De-duplicate and sort
            seen = set()
            unique_word_hyps: List[Tuple[List[str], float]] = []
            for words, score in sorted(
                word_hyps_list, key=lambda x: x[1], reverse=True
            ):
                key = tuple(words)
                if key not in seen:
                    seen.add(key)
                    unique_word_hyps.append((words, score))

            result.word_hypotheses = unique_word_hyps[: self.config.max_word_candidates]
            if unique_word_hyps:
                result.best_words = unique_word_hyps[0][0]
            result.lexicon_time = time.time() - t_lex

        # Uncertainty estimation
        t_unc = time.time()
        frame_info = self.uncertainty.compute_frame_uncertainty(log_probs, length)
        result.frame_uncertainty = frame_info

        if result.best_words and result.phoneme_hypotheses:
            best_hyp = result.phoneme_hypotheses[0]
            if best_hyp.label_posteriors is not None:
                word_confs = self.uncertainty.compute_word_confidence_from_posteriors(
                    frame_info,
                    best_hyp.label_posteriors,
                    result.best_phoneme_ids,
                    result.best_words,
                )
            else:
                word_spans = self._estimate_word_spans(
                    frame_info, result.best_phoneme_ids, result.best_words
                )
                word_confs = self.uncertainty.compute_word_confidence(
                    frame_info, word_spans
                )
            result.word_confidences = word_confs
        result.uncertainty_time = time.time() - t_unc

        # B2 + C: Legacy LLM-based modes
        t_llm = time.time()
        if result.word_hypotheses and mode != "neural_only":
            word_conf_tuples = [
                (wc.word, wc.confidence) for wc in result.word_confidences
            ]

            template = self.hyp_builder.build(result.word_hypotheses, word_conf_tuples)
            result.template = template

            if mode == "student" and self.config.use_student:
                student = self._get_student()
                cand_lists = [wh[0] for wh in result.word_hypotheses[:20]]
                neural_scores = [wh[1] for wh in result.word_hypotheses[:20]]

                feats = []
                for i, (words, ns) in enumerate(zip(cand_lists, neural_scores)):
                    feat = CandidateFeatures(
                        neural_score=ns,
                        n_words=len(words),
                        n_uncertain_slots=template.n_open_slots,
                        n_locked_slots=template.n_locked_slots,
                    )
                    feats.append(feat)

                best_words, best_score = student.rerank(cand_lists, feats)
                result.final_words = best_words
                result.final_score = best_score

            elif mode == "slot_filling":
                llm = self._get_llm()
                slot_decoder = SlotFillingDecoder(
                    llm_score_fn=llm.score,
                    lambda_neural=self.config.lambda_neural,
                    lambda_lm=self.config.lambda_lm,
                    gamma_constraint=self.config.gamma_constraint,
                    beam_width=self.config.slot_filling_beam,
                )
                best_words, best_score = slot_decoder.decode(template)
                result.final_words = best_words
                result.final_score = best_score

            elif mode == "constrained_rescore":
                llm = self._get_llm()
                slot_decoder = SlotFillingDecoder(
                    llm_score_fn=llm.score,
                    lambda_neural=self.config.lambda_neural,
                    lambda_lm=self.config.lambda_lm,
                    gamma_constraint=self.config.gamma_constraint,
                )
                n_cands = min(
                    len(result.word_hypotheses), self.config.max_word_candidates
                )
                cand_lists = [wh[0] for wh in result.word_hypotheses[:n_cands]]
                neural_scores = [wh[1] for wh in result.word_hypotheses[:n_cands]]
                sentences = [" ".join(words) for words in cand_lists]

                if self._ngram_lm is not None and self._ngram_lm.is_ready:
                    ngram_scores = [self._ngram_lm.score_sentence(s) for s in sentences]
                else:
                    ngram_scores = None

                lm_scores = llm.score_batch(sentences)
                rescored = slot_decoder.rescore_nbest(
                    template,
                    cand_lists,
                    neural_scores,
                    length_penalty_beta=self.config.length_penalty_beta,
                    lm_scores=lm_scores,
                    ngram_scores=ngram_scores,
                    lambda_ngram=self.config.lambda_ngram,
                    normalize_scores=self.config.normalize_scores,
                )
                if rescored:
                    result.final_words = rescored[0][0]
                    result.final_score = rescored[0][1]

            elif mode == "unconstrained_rescore":
                llm = self._get_llm()
                n_cands = min(
                    len(result.word_hypotheses), self.config.max_word_candidates
                )
                cand_lists = [wh[0] for wh in result.word_hypotheses[:n_cands]]
                neural_scores_list = [wh[1] for wh in result.word_hypotheses[:n_cands]]
                sentences = [" ".join(words) for words in cand_lists]

                if self._ngram_lm is not None and self._ngram_lm.is_ready:
                    ngram_scores = [self._ngram_lm.score_sentence(s) for s in sentences]
                else:
                    ngram_scores = [0.0] * n_cands

                lm_scores = llm.score_batch(sentences)

                ns_arr = np.array(neural_scores_list, dtype=np.float64)
                ng_arr = np.array(ngram_scores, dtype=np.float64)
                lm_arr = np.array(lm_scores, dtype=np.float64)
                if self.config.normalize_scores and len(ns_arr) > 1:
                    for arr in [ns_arr, ng_arr, lm_arr]:
                        std = np.std(arr)
                        if std > 1e-8:
                            arr -= np.mean(arr)
                            arr /= std
                        else:
                            arr -= np.mean(arr)

                rescored = []
                for i, words in enumerate(cand_lists):
                    combined = (
                        self.config.lambda_neural * ns_arr[i]
                        + self.config.lambda_ngram * ng_arr[i]
                        + self.config.lambda_lm * lm_arr[i]
                        + self.config.length_penalty_beta * len(words)
                    )
                    rescored.append((words, combined))
                rescored.sort(key=lambda x: x[1], reverse=True)

                if rescored:
                    result.final_words = rescored[0][0]
                    result.final_score = rescored[0][1]

            else:
                result.final_words = result.best_words
                result.final_score = (
                    result.word_hypotheses[0][1] if result.word_hypotheses else 0.0
                )

        elif result.best_words:
            result.final_words = result.best_words
            result.final_score = (
                result.word_hypotheses[0][1] if result.word_hypotheses else 0.0
            )

        # For neural_only, just use the best words
        if mode == "neural_only" and not result.final_words and result.best_words:
            result.final_words = result.best_words
            result.final_score = (
                result.word_hypotheses[0][1] if result.word_hypotheses else 0.0
            )

        result.llm_time = time.time() - t_llm
        result.final_sentence = " ".join(result.final_words)
        result.total_time = time.time() - t0
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_word_spans(
        self,
        frame_info: FrameUncertainty,
        phoneme_ids: List[int],
        words: List[str],
    ) -> List[Tuple[str, int, int]]:
        """
        Estimate word-level frame spans by splitting phoneme IDs on SIL
        and distributing frames proportionally.
        """
        if not words or not phoneme_ids:
            return []

        chunks: List[List[int]] = []
        current: List[int] = []
        for pid in phoneme_ids:
            if pid == SIL_IDX:
                if current:
                    chunks.append(current)
                    current = []
            else:
                current.append(pid)
        if current:
            chunks.append(current)

        n_phones_total = sum(len(c) for c in chunks)
        if n_phones_total == 0:
            T = frame_info.length
            span = T // max(len(words), 1)
            return [(w, i * span, (i + 1) * span) for i, w in enumerate(words)]

        T = frame_info.length
        spans: List[Tuple[str, int, int]] = []
        frame_pos = 0

        for i, word in enumerate(words):
            if i < len(chunks):
                n_phones_chunk = len(chunks[i])
            else:
                n_phones_chunk = 1

            frame_span = max(1, int(T * n_phones_chunk / n_phones_total))
            start = frame_pos
            end = min(frame_pos + frame_span, T)
            spans.append((word, start, end))
            frame_pos = end

        return spans

    def collect_distillation_data(
        self,
        log_probs_list: List[np.ndarray],
        lengths: List[Optional[int]],
    ) -> DistillationDataset:
        """D1 – Collect distillation data from a set of utterances."""
        llm = self._get_llm()
        collector = DistillationDataCollector(
            llm_score_fn=llm.score,
            lambda_neural=self.config.lambda_neural,
            lambda_lm=self.config.lambda_lm,
        )

        dataset = DistillationDataset()

        for log_probs, length_val in zip(log_probs_list, lengths):
            result = self.decode_utterance(log_probs, length_val)

            if not result.word_hypotheses:
                continue

            cand_lists = [wh[0] for wh in result.word_hypotheses[:20]]
            neural_scores = [wh[1] for wh in result.word_hypotheses[:20]]

            unc_feats = []
            for words, ns in zip(cand_lists, neural_scores):
                template = result.template
                uf = {
                    "mean_entropy": float(np.mean(result.frame_uncertainty.entropy))
                    if result.frame_uncertainty
                    else 0.0,
                    "min_margin": float(np.min(result.frame_uncertainty.margin))
                    if result.frame_uncertainty
                    else 0.0,
                    "blank_ratio": float(
                        np.mean(result.frame_uncertainty.argmax == BLANK_IDX)
                    )
                    if result.frame_uncertainty
                    else 0.0,
                    "n_uncertain_slots": template.n_open_slots if template else 0,
                    "n_locked_slots": template.n_locked_slots if template else 0,
                    "edit_distance_to_best": 0,
                }
                unc_feats.append(uf)

            sample = collector.collect_sample(cand_lists, neural_scores, unc_feats)
            dataset.samples.append(sample)

        return dataset


def _longest_common_prefix(word_lists: List[List[str]]) -> List[str]:
    """Find the longest common word prefix across multiple word sequences."""
    if not word_lists:
        return []
    prefix = []
    for i in range(min(len(wl) for wl in word_lists)):
        words_at_i = {wl[i] for wl in word_lists}
        if len(words_at_i) == 1:
            prefix.append(word_lists[0][i])
        else:
            break
    return prefix
