"""
Decoding Pipeline
==================
Orchestrates the full decoding pipeline:

  Conformer → logits → rearrange → WFST n-gram decode → n-best augment → OPT rescore

This follows the NEJM brain-to-text pipeline (Card et al., NEJM 2024):

1. Conformer produces CTC logits [T, 41] in model order
   [BLANK, phonemes(1-39), SIL(40)].
2. Logits are rearranged to Kaldi order [BLANK, SIL, phonemes(1-39)].
3. WFST decoder (C++ lm_decoder) performs CTC beam search with
   n-gram LM integration, producing n-best word hypotheses.
4. Optionally rescore with unpruned n-gram (lattice rescoring).
5. N-best list is augmented by word swapping between top candidates.
6. OPT/LLM rescores the augmented n-best to pick the final output.

Usage
-----
    from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig
    config = PipelineConfig(lm_path="/path/to/lm")
    pipeline = DecodePipeline(config)
    result = pipeline.decode(logits)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ngram_decoder import NgramWFSTDecoder, NgramDecodeResult, rearrange_logits
from .nbest_augmentation import augment_nbest

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full decoding pipeline."""

    # ---- WFST N-gram decoder ----
    lm_path: str = ""  # Path to directory with TLG.fst, words.txt, etc.
    max_active: int = 7000
    min_active: int = 200
    beam: float = 17.0
    lattice_beam: float = 8.0
    acoustic_scale: float = 0.3
    ctc_blank_skip_threshold: float = 1.0
    length_penalty: float = 0.0
    nbest: int = 100
    blank_penalty: float = 9.0
    rescore: bool = False  # Rescore with unpruned G.fst (requires G_no_prune.fst)

    # ---- N-best augmentation ----
    augment_nbest: bool = True
    top_candidates_to_augment: int = 20
    score_penalty_percent: float = 0.01

    # ---- Input format ----
    is_log_probs: bool = False  # True if model outputs log_softmax (e.g. Conformer)

    # ---- LLM rescoring ----
    do_llm: bool = True
    llm_model_name: str = "facebook/opt-6.7b"
    llm_cache_dir: Optional[str] = None
    llm_device: str = "cuda"
    llm_alpha: float = 0.5  # Higher = more weight on LLM, lower = more on n-gram
    llm_length_penalty: float = 0.0


@dataclass
class DecodeResult:
    """Result from decoding one utterance."""

    # Best hypothesis
    sentence: str = ""
    confidence: float = 0.0

    # N-best list (after all rescoring)
    # Each entry: (sentence, ac_score, ngram_score, llm_score, total_score)
    nbest: List[Dict] = field(default_factory=list)

    # Raw outputs from each stage
    ngram_nbest: List[NgramDecodeResult] = field(default_factory=list)
    augmented_nbest: List[Tuple[str, float, float]] = field(default_factory=list)

    # Timing
    ngram_time: float = 0.0
    augment_time: float = 0.0
    llm_time: float = 0.0
    total_time: float = 0.0


class DecodePipeline:
    """
    Full decoding pipeline: WFST n-gram + n-best augmentation + LLM rescore.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # WFST decoder (initialized lazily or explicitly)
        self._ngram_decoder: Optional[NgramWFSTDecoder] = None

        # LLM rescorer (initialized lazily)
        self._llm_rescorer = None

    def setup(self):
        """Initialize the WFST decoder. Must be called before decode()."""
        if not self.config.lm_path:
            raise ValueError(
                "lm_path must be set in PipelineConfig before calling setup(). "
                "Point it to a directory containing TLG.fst and words.txt."
            )

        self._ngram_decoder = NgramWFSTDecoder(
            lm_path=self.config.lm_path,
            max_active=self.config.max_active,
            min_active=self.config.min_active,
            beam=self.config.beam,
            lattice_beam=self.config.lattice_beam,
            acoustic_scale=self.config.acoustic_scale,
            ctc_blank_skip_threshold=self.config.ctc_blank_skip_threshold,
            length_penalty=self.config.length_penalty,
            nbest=self.config.nbest,
        )
        logger.info("Decode pipeline initialized")

    def _get_llm(self):
        """Lazy-load the LLM rescorer."""
        if self._llm_rescorer is None:
            from .llm_rescorer import LLMRescorer

            self._llm_rescorer = LLMRescorer(
                model_name=self.config.llm_model_name,
                cache_dir=self.config.llm_cache_dir,
                device=self.config.llm_device,
            )
        return self._llm_rescorer

    def update_params(self, **kwargs):
        """
        Update decoder parameters at runtime.

        Accepts any PipelineConfig field. WFST decoder params are
        forwarded to the C++ decoder.
        """
        wfst_params = {
            "max_active",
            "min_active",
            "beam",
            "lattice_beam",
            "acoustic_scale",
            "ctc_blank_skip_threshold",
            "length_penalty",
            "nbest",
        }

        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        # Forward WFST params to the C++ decoder
        if self._ngram_decoder is not None and wfst_params & set(kwargs.keys()):
            self._ngram_decoder.update_params(
                max_active=self.config.max_active,
                min_active=self.config.min_active,
                beam=self.config.beam,
                lattice_beam=self.config.lattice_beam,
                acoustic_scale=self.config.acoustic_scale,
                ctc_blank_skip_threshold=self.config.ctc_blank_skip_threshold,
                length_penalty=self.config.length_penalty,
                nbest=self.config.nbest,
            )

    def decode(
        self,
        logits: np.ndarray,
        already_rearranged: bool = False,
        context_str: Optional[str] = None,
    ) -> DecodeResult:
        """
        Decode a single utterance through the full pipeline.

        Parameters
        ----------
        logits : np.ndarray, shape [T, 41]
            Raw logits from the Conformer.
            Model order: [BLANK, phonemes(1-39), SIL(40)].
        already_rearranged : bool
            If True, logits are already in Kaldi order and won't be rearranged.
        context_str : str or None
            Optional context for LLM rescoring (e.g. previously decoded text).

        Returns
        -------
        DecodeResult
        """
        if self._ngram_decoder is None:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        result = DecodeResult()
        t0 = time.time()

        # Step 1: Rearrange logits to Kaldi order
        if not already_rearranged:
            logits = rearrange_logits(logits)

        # Step 2: WFST n-gram decoding
        t_ngram = time.time()
        ngram_results = self._ngram_decoder.decode(
            logits,
            blank_penalty=self.config.blank_penalty,
            rescore=self.config.rescore,
            is_log_probs=self.config.is_log_probs,
        )
        result.ngram_nbest = ngram_results
        result.ngram_time = time.time() - t_ngram

        if not ngram_results:
            logger.warning("No output from WFST decoder")
            result.total_time = time.time() - t0
            return result

        # Convert to (sentence, ac_score, lm_score) tuples
        nbest_tuples = [(r.sentence, r.ac_score, r.lm_score) for r in ngram_results]

        # Step 3: N-best augmentation
        if self.config.augment_nbest and self.config.nbest > 1:
            t_aug = time.time()
            n_before = len(nbest_tuples)
            nbest_tuples = augment_nbest(
                nbest_tuples,
                top_candidates_to_augment=self.config.top_candidates_to_augment,
                acoustic_scale=self.config.acoustic_scale,
                score_penalty_percent=self.config.score_penalty_percent,
            )
            result.augmented_nbest = nbest_tuples
            result.augment_time = time.time() - t_aug
            logger.debug(
                f"Augmented n-best from {n_before} to {len(nbest_tuples)} candidates"
            )
        else:
            result.augmented_nbest = nbest_tuples

        # Step 4: LLM rescoring
        if self.config.do_llm and self.config.nbest > 1:
            t_llm = time.time()
            llm = self._get_llm()
            best_sentence, nbest_out, confidence = llm.rescore(
                nbest_tuples,
                acoustic_scale=self.config.acoustic_scale,
                alpha=self.config.llm_alpha,
                length_penalty=self.config.llm_length_penalty,
                context_str=context_str,
                return_confidence=True,
            )
            result.sentence = best_sentence
            result.confidence = confidence
            result.llm_time = time.time() - t_llm

            # Parse nbest_out into structured format
            for entry in nbest_out:
                parts = entry.split(";")
                if len(parts) >= 5:
                    result.nbest.append(
                        {
                            "sentence": parts[0],
                            "ac_score": float(parts[1]),
                            "ngram_score": float(parts[2]),
                            "llm_score": float(parts[3]),
                            "total_score": float(parts[4]),
                        }
                    )

        elif ngram_results:
            # No LLM — just use the best n-gram result
            result.sentence = ngram_results[0].sentence
            result.confidence = 1.0
            for sent, ac, lm in nbest_tuples:
                result.nbest.append(
                    {
                        "sentence": sent,
                        "ac_score": ac,
                        "ngram_score": lm,
                        "llm_score": 0.0,
                        "total_score": self.config.acoustic_scale * ac + lm,
                    }
                )

        result.total_time = time.time() - t0
        return result

    def decode_batch(
        self,
        logits_list: List[np.ndarray],
        already_rearranged: bool = False,
        context_str: Optional[str] = None,
    ) -> List[DecodeResult]:
        """
        Decode a batch of utterances.

        Parameters
        ----------
        logits_list : list of np.ndarray
            Each element is [T_i, 41] logits for one utterance.
        already_rearranged : bool
            If True, logits are already in Kaldi order.
        context_str : str or None
            Context string for LLM rescoring.

        Returns
        -------
        list of DecodeResult
        """
        results = []
        for logits in logits_list:
            result = self.decode(
                logits,
                already_rearranged=already_rearranged,
                context_str=context_str,
            )
            results.append(result)
        return results
