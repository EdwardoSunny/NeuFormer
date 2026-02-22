"""
LLM Rescorer
=============
OPT-style LLM rescoring for n-best hypothesis lists, ported from
the NEJM brain-to-text repo's ``language-model-standalone.py``.

Supports any HuggingFace causal LM (OPT-6.7B, GPT-2, Llama, etc.).
Scores hypotheses by computing per-token log-probabilities and
combines them with acoustic + n-gram scores.

Usage
-----
    from neural_decoder.llm_rescorer import LLMRescorer
    rescorer = LLMRescorer(model_name="facebook/opt-6.7b")
    best, nbest_out = rescorer.rescore(
        nbest_hypotheses,
        acoustic_scale=0.3,
        alpha=0.5,
    )
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LLMRescorer:
    """
    LLM-based n-best rescorer.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. "facebook/opt-6.7b", "gpt2").
    cache_dir : str or None
        Directory for caching model weights.
    device : str
        Device for inference ("cuda:0", "cpu", etc.).
    torch_dtype : torch.dtype
        Model precision. Default float16 for OPT (requires ~13GB VRAM).
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-6.7b",
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """Lazy-load the model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading LLM: {self.model_name} ...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
        )

        if self.device.type != "cpu":
            self._model = self._model.to(self.device)

        self._model.eval()

        # Ensure padding token
        self._tokenizer.padding_side = "right"
        self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"LLM loaded on {self.device}")

    @torch.inference_mode()
    def score_batch(
        self,
        hypotheses: List[str],
        length_penalty: float = 0.0,
    ) -> np.ndarray:
        """
        Score a batch of hypotheses.

        Parameters
        ----------
        hypotheses : list of str
            Candidate sentences.
        length_penalty : float
            Per-token penalty (subtracted from total score).

        Returns
        -------
        np.ndarray, shape [N]
            Log-probability scores for each hypothesis.
        """
        self._ensure_loaded()

        if not hypotheses:
            return np.array([])

        self._model.eval()

        try:
            # Try scoring all at once
            scores = self._score_batch_impl(hypotheses, length_penalty)
        except Exception as e:
            logger.warning(f"Batch scoring failed ({e}), falling back to chunked")
            # Fall back to chunked scoring
            scores = []
            chunk_size = max(1, len(hypotheses) // 5)
            for i in range(0, len(hypotheses), chunk_size):
                chunk = hypotheses[i : i + chunk_size]
                try:
                    chunk_scores = self._score_batch_impl(chunk, length_penalty)
                    scores.extend(chunk_scores)
                except Exception as e2:
                    logger.error(f"Chunk scoring also failed: {e2}")
                    scores.extend([0.0] * len(chunk))
            scores = np.array(scores)

        return scores

    def _score_batch_impl(
        self,
        hypotheses: List[str],
        length_penalty: float,
    ) -> np.ndarray:
        """Internal batch scoring implementation."""
        inputs = self._tokenizer(hypotheses, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        log_probs = log_probs.cpu().numpy()

        input_ids = inputs["input_ids"].cpu().numpy()
        attention_mask = inputs["attention_mask"].cpu().numpy()
        batch_size, seq_len, _ = log_probs.shape

        scores = []
        for i in range(batch_size):
            n_tokens = int(attention_mask[i].sum())
            # Sum log-probs of each token given previous context
            score = sum(
                log_probs[i, t - 1, input_ids[i, t]] for t in range(1, n_tokens)
            )
            scores.append(score - n_tokens * length_penalty)

        return np.array(scores)

    def rescore(
        self,
        nbest: List[Tuple[str, float, float]],
        acoustic_scale: float = 0.3,
        alpha: float = 0.5,
        length_penalty: float = 0.0,
        context_str: Optional[str] = None,
        return_confidence: bool = False,
    ) -> Tuple[str, List[str], float] | Tuple[str, List[str]]:
        """
        Rescore an n-best list using the LLM.

        This is the core rescoring function from the NEJM pipeline.
        The final score for each hypothesis is:
            total = acoustic_scale * ac_score + (1 - alpha) * ngram_score + alpha * llm_score

        Parameters
        ----------
        nbest : list of (sentence, ac_score, lm_score)
            N-best hypotheses from the WFST decoder.
            sentence: str, ac_score: float, lm_score: float (n-gram score).
        acoustic_scale : float
            Weight for acoustic scores.
        alpha : float
            Interpolation weight [0-1]. Higher = more weight on LLM.
        length_penalty : float
            Per-token penalty for LLM scoring.
        context_str : str or None
            Optional context string prepended to each hypothesis for
            contextual LLM scoring.
        return_confidence : bool
            If True, also return the confidence (softmax probability)
            of the best hypothesis.

        Returns
        -------
        best_sentence : str
            Best hypothesis after rescoring.
        nbest_out : list of str
            Detailed scoring info for each hypothesis (semicolon-separated).
        confidence : float (only if return_confidence=True)
            Confidence of the best hypothesis.
        """
        hypotheses = []
        acoustic_scores = []
        old_lm_scores = []

        for out in nbest:
            hyp = out[0].strip()
            if len(hyp) == 0:
                continue

            # Add context to the front of each sentence
            if context_str is not None and len(context_str.split()) > 0:
                hyp = context_str + " " + hyp

            # Clean up punctuation spacing
            hyp = hyp.replace(">", "")
            hyp = hyp.replace("  ", " ")
            hyp = hyp.replace(" ,", ",")
            hyp = hyp.replace(" .", ".")
            hyp = hyp.replace(" ?", "?")

            hypotheses.append(hyp)
            acoustic_scores.append(out[1])
            old_lm_scores.append(out[2])

        if len(hypotheses) == 0:
            logger.error("No valid hypotheses to rescore")
            if return_confidence:
                return "", [], 0.0
            return "", []

        acoustic_scores = np.array(acoustic_scores)
        old_lm_scores = np.array(old_lm_scores)

        # Get new LM scores
        new_lm_scores = self.score_batch(hypotheses, length_penalty)

        # Remove context from hypotheses for output
        if context_str is not None and len(context_str.split()) > 0:
            hypotheses = [h[len(context_str) + 1 :] for h in hypotheses]

        # Calculate total scores
        total_scores = (
            acoustic_scale * acoustic_scores
            + (1 - alpha) * old_lm_scores
            + alpha * new_lm_scores
        )

        # Get the best hypothesis
        max_idx = np.argmax(total_scores)
        best_hyp = hypotheses[max_idx]

        # Create nbest output
        nbest_out = []
        min_len = min(len(nbest), len(new_lm_scores), len(total_scores))
        for i in range(min_len):
            nbest_out.append(
                ";".join(
                    map(
                        str,
                        [
                            nbest[i][0],
                            nbest[i][1],
                            nbest[i][2],
                            new_lm_scores[i],
                            total_scores[i],
                        ],
                    )
                )
            )

        if not return_confidence:
            return best_hyp, nbest_out
        else:
            total_scores_shifted = total_scores - np.max(total_scores)
            probs = np.exp(total_scores_shifted)
            confidence = probs[max_idx] / np.sum(probs)
            return best_hyp, nbest_out, confidence
