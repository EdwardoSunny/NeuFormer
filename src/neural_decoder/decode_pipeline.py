"""
Full Posterior-Constrained Decoding Pipeline
=============================================
Orchestrates the entire decoding pipeline:
  A1. CTC beam search → phoneme N-best
  A2. Phoneme → word conversion via lexicon
  B1. Uncertainty estimation
  B2. Constrained hypothesis building
  C.  LLM rescoring / slot-filling
  D.  Optional student reranking

Usage
-----
    from neural_decoder.decode_pipeline import DecodePipeline
    pipe = DecodePipeline(config)
    pipe.setup(training_sentences)
    result = pipe.decode_utterance(log_probs, length)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ctc_beam_decoder import CTCBeamDecoder, CTCHypothesis
from .lexicon import PronunciationLexicon
from .uncertainty import UncertaintyEstimator, FrameUncertainty, WordConfidence
from .constrained_decode import (
    ConstrainedHypothesisBuilder,
    ConstrainedTemplate,
    SlotFillingDecoder,
)
from .llm_scorer import LLMScorer
from .distillation import (
    CandidateFeatures,
    DistillationDataCollector,
    DistillationDataset,
    DistillationSample,
    StudentReranker,
)
from .phoneme_table import BLANK_IDX, SIL_IDX


@dataclass
class PipelineConfig:
    """Configuration for the full decoding pipeline."""

    # A1 – CTC beam search
    beam_width: int = 40
    n_best: int = 25
    blank_penalty: float = 0.0

    # A2 – Lexicon
    max_edit_dist: int = 1
    lexicon_beam_width: int = 5

    # B1 – Uncertainty
    entropy_threshold: float = 1.5
    margin_threshold: float = 0.3

    # B2 – Constrained hypothesis
    high_confidence_threshold: float = 0.85
    max_candidates_per_slot: int = 20

    # C – LLM
    llm_model_name: str = "meta-llama/Meta-Llama-3-8B"
    llm_device: str = "cpu"
    llm_load_in_8bit: bool = False
    lambda_neural: float = 0.5
    lambda_lm: float = 1.0
    gamma_constraint: float = 2.0
    length_penalty_beta: float = 0.0
    slot_filling_beam: int = 10

    # D – Student
    use_student: bool = False
    student_model_path: Optional[str] = None

    # Mode
    decode_mode: str = "constrained_rescore"  # or "slot_filling" or "student"


@dataclass
class DecodeResult:
    """Result from decoding one utterance."""

    # Phoneme level
    phoneme_hypotheses: List[CTCHypothesis] = field(default_factory=list)
    best_phoneme_ids: List[int] = field(default_factory=list)
    phoneme_log_prob: float = 0.0

    # Word level
    word_hypotheses: List[Tuple[List[str], float]] = field(default_factory=list)
    best_words: List[str] = field(default_factory=list)

    # Uncertainty
    frame_uncertainty: Optional[FrameUncertainty] = None
    word_confidences: List[WordConfidence] = field(default_factory=list)

    # Constrained
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
    Full posterior-constrained decoding pipeline.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # A1
        self.ctc_decoder = CTCBeamDecoder(
            beam_width=config.beam_width,
            blank=BLANK_IDX,
            n_best=config.n_best,
            blank_penalty=config.blank_penalty,
        )

        # A2
        self.lexicon = PronunciationLexicon(
            max_edit_dist=config.max_edit_dist,
            sil_idx=SIL_IDX,
        )

        # B1
        self.uncertainty = UncertaintyEstimator(
            blank_idx=BLANK_IDX,
            sil_idx=SIL_IDX,
            entropy_threshold=config.entropy_threshold,
            margin_threshold=config.margin_threshold,
        )

        # B2
        self.hyp_builder = ConstrainedHypothesisBuilder(
            high_confidence_threshold=config.high_confidence_threshold,
            max_candidates_per_slot=config.max_candidates_per_slot,
        )

        # C – LLM (lazy-loaded)
        self._llm_scorer: Optional[LLMScorer] = None

        # D – Student (lazy-loaded)
        self._student: Optional[StudentReranker] = None

    def setup(self, training_sentences: List[str]):
        """Build the lexicon from training data."""
        self.lexicon.build_from_sentences(training_sentences)

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
        result = DecodeResult()
        t0 = time.time()

        # ---- A1: CTC beam search ----
        t_ctc = time.time()
        hyps = self.ctc_decoder.decode(log_probs, length)
        result.phoneme_hypotheses = hyps
        if hyps:
            result.best_phoneme_ids = hyps[0].phoneme_ids
            result.phoneme_log_prob = hyps[0].log_prob
        result.ctc_decode_time = time.time() - t_ctc

        # ---- A2: Phoneme → Word ----
        t_lex = time.time()
        word_hyps: List[Tuple[List[str], float]] = []
        T_frames = (
            log_probs.shape[0] if length is None else min(log_probs.shape[0], length)
        )
        for hyp in hyps:
            w_results = self.lexicon.phonemes_to_words(
                hyp.phoneme_ids, beam_width=self.config.lexicon_beam_width
            )
            # Normalize CTC log-prob per frame so it's comparable to
            # per-token LLM scores (both will be in ~ [-1, -10] range)
            norm_ctc = hyp.log_prob / max(T_frames, 1)
            for words, edit_cost in w_results:
                # Combine normalized CTC score with edit cost penalty
                combined = norm_ctc - edit_cost
                word_hyps.append((words, combined))

        # De-duplicate and sort
        seen = set()
        unique_word_hyps: List[Tuple[List[str], float]] = []
        for words, score in sorted(word_hyps, key=lambda x: x[1], reverse=True):
            key = tuple(words)
            if key not in seen:
                seen.add(key)
                unique_word_hyps.append((words, score))

        result.word_hypotheses = unique_word_hyps[: self.config.n_best * 5]
        if unique_word_hyps:
            result.best_words = unique_word_hyps[0][0]
        result.lexicon_time = time.time() - t_lex

        # ---- B1: Uncertainty estimation ----
        t_unc = time.time()
        frame_info = self.uncertainty.compute_frame_uncertainty(log_probs, length)
        result.frame_uncertainty = frame_info

        # Compute word-level confidence for the best hypothesis
        if result.best_words and hyps:
            best_hyp = hyps[0]
            # Use forward-backward posteriors if available (from forced alignment)
            if best_hyp.label_posteriors is not None:
                word_confs = self.uncertainty.compute_word_confidence_from_posteriors(
                    frame_info,
                    best_hyp.label_posteriors,
                    result.best_phoneme_ids,
                    result.best_words,
                )
            else:
                # Fallback to proportional span estimation
                word_spans = self._estimate_word_spans(
                    frame_info, result.best_phoneme_ids, result.best_words
                )
                word_confs = self.uncertainty.compute_word_confidence(
                    frame_info, word_spans
                )
            result.word_confidences = word_confs
        result.uncertainty_time = time.time() - t_unc

        # ---- B2 + C: Constrained LLM decoding ----
        t_llm = time.time()
        if result.word_hypotheses and self.config.decode_mode != "neural_only":
            # Build confidence tuples for the builder
            word_conf_tuples = [
                (wc.word, wc.confidence) for wc in result.word_confidences
            ]

            template = self.hyp_builder.build(result.word_hypotheses, word_conf_tuples)
            result.template = template

            if self.config.decode_mode == "student" and self.config.use_student:
                # D3 – Student reranking
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

            elif self.config.decode_mode == "slot_filling":
                # C2 – Slot-filling
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

            elif self.config.decode_mode == "constrained_rescore":
                # C1 – Constrained N-best rescoring
                llm = self._get_llm()
                slot_decoder = SlotFillingDecoder(
                    llm_score_fn=llm.score,
                    lambda_neural=self.config.lambda_neural,
                    lambda_lm=self.config.lambda_lm,
                    gamma_constraint=self.config.gamma_constraint,
                )
                cand_lists = [wh[0] for wh in result.word_hypotheses[:50]]
                neural_scores = [wh[1] for wh in result.word_hypotheses[:50]]
                rescored = slot_decoder.rescore_nbest(
                    template,
                    cand_lists,
                    neural_scores,
                    length_penalty_beta=self.config.length_penalty_beta,
                )
                if rescored:
                    result.final_words = rescored[0][0]
                    result.final_score = rescored[0][1]

            elif self.config.decode_mode == "unconstrained_rescore":
                # Plain LLM rescoring (no constraint penalties)
                llm = self._get_llm()
                cand_lists = [wh[0] for wh in result.word_hypotheses[:50]]
                neural_scores = [wh[1] for wh in result.word_hypotheses[:50]]

                rescored = []
                for words, ns in zip(cand_lists, neural_scores):
                    sentence = " ".join(words)
                    lm_score = llm.score(sentence)
                    combined = (
                        self.config.lambda_neural * ns
                        + self.config.lambda_lm * lm_score
                    )
                    rescored.append((words, combined))
                rescored.sort(key=lambda x: x[1], reverse=True)
                if rescored:
                    result.final_words = rescored[0][0]
                    result.final_score = rescored[0][1]

            else:
                # neural_only – just use best word hypothesis
                result.final_words = result.best_words
                result.final_score = (
                    result.word_hypotheses[0][1] if result.word_hypotheses else 0.0
                )

        elif result.best_words:
            result.final_words = result.best_words
            result.final_score = (
                result.word_hypotheses[0][1] if result.word_hypotheses else 0.0
            )

        result.llm_time = time.time() - t_llm
        result.final_sentence = " ".join(result.final_words)
        result.total_time = time.time() - t0

        return result

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

        # Split phonemes into word chunks at SIL boundaries
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

        # Match chunks to words (best effort)
        n_phones_total = sum(len(c) for c in chunks)
        if n_phones_total == 0:
            # Distribute evenly
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

            # Proportional frame allocation
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
        """
        D1 – Collect distillation data from a set of utterances.

        Runs the full pipeline on each utterance, then uses the
        LLM teacher to score all candidates.
        """
        llm = self._get_llm()
        collector = DistillationDataCollector(
            llm_score_fn=llm.score,
            lambda_neural=self.config.lambda_neural,
            lambda_lm=self.config.lambda_lm,
        )

        dataset = DistillationDataset()

        for log_probs, length in zip(log_probs_list, lengths):
            result = self.decode_utterance(log_probs, length)

            if not result.word_hypotheses:
                continue

            cand_lists = [wh[0] for wh in result.word_hypotheses[:20]]
            neural_scores = [wh[1] for wh in result.word_hypotheses[:20]]

            # Build uncertainty features per candidate
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
