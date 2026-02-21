"""
C1 – LLM Scoring Module
========================
Wraps a HuggingFace causal LM (e.g. GPT-2, Llama 3) to provide
length-normalised log-probability scoring of text sequences.

This replaces the old OPT-6B + 5-gram pipeline with a clean,
configurable scorer that can be used by both the constrained-rescore
(C1) and slot-filling (C2) decoders.

Key features:
  * Whole-sequence scoring with length normalisation
  * **True incremental scoring**: caches prefix KV states and scores
    only continuation tokens, with proper length normalisation on
    continuation tokens only (avoids re-scoring prefix and bias toward
    shorter continuations)

Usage
-----
    from neural_decoder.llm_scorer import LLMScorer
    scorer = LLMScorer(model_name="meta-llama/Meta-Llama-3-8B")
    score = scorer.score("hello world")
    # Incremental: score only the continuation tokens
    scores = scorer.score_incremental("the cat", ["sat on", "flew to"])
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import numpy as np


class LLMScorer:
    """
    Language model scorer backed by a HuggingFace causal LM.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. "meta-llama/Meta-Llama-3-8B",
        "gpt2").
    device : str
        Device string ("cuda", "cpu", "auto").
    load_in_8bit : bool
        Whether to load the model in 8-bit quantisation (requires
        bitsandbytes).
    max_length : int
        Maximum token length for scoring.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: str = "auto",
        load_in_8bit: bool = False,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs = {}
        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        elif self.device == "auto":
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )

        if not self.load_in_8bit and self.device != "auto":
            self._model = self._model.to(self.device)

        self._model.eval()

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    def _get_device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self._model.parameters()).device

    @torch.no_grad()
    def score(self, text: str) -> float:
        """
        Compute length-normalised log-probability of *text*.

        Returns
        -------
        float
            (1 / n_tokens) * sum_t log P(token_t | prefix)
        """
        self._ensure_loaded()

        if not text.strip():
            return 0.0

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        device = self._get_device()
        input_ids = inputs["input_ids"].to(device)

        if input_ids.shape[1] <= 1:
            return 0.0

        outputs = self._model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

        # Shift: logits[t] predicts input_ids[t+1]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(
            -1
        )  # [1, seq_len-1]

        total_log_prob = token_log_probs.sum().item()
        n_tokens = token_log_probs.shape[1]

        return total_log_prob / max(n_tokens, 1)

    @torch.no_grad()
    def score_batch(self, texts: List[str], batch_size: int = 16) -> List[float]:
        """
        Score a batch of texts with true batched forward passes.

        Pads sequences to equal length within each mini-batch and runs
        a single forward pass per mini-batch, which is much faster than
        scoring one at a time on GPU.

        Returns list of normalised log-probs.
        """
        self._ensure_loaded()

        if not texts:
            return []

        device = self._get_device()
        all_scores: List[float] = [0.0] * len(texts)

        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not valid_indices:
            return all_scores

        for start in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[start : start + batch_size]
            batch_texts = [texts[i] for i in batch_indices]

            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            if input_ids.shape[1] <= 1:
                continue

            outputs = self._model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, seq_len, vocab_size]

            # Shift: logits[t] predicts input_ids[t+1]
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]  # mask for target tokens

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(
                -1
            )  # [B, seq_len-1]

            # Mask out padding tokens
            token_log_probs = token_log_probs * shift_mask

            for j, idx in enumerate(batch_indices):
                n_tokens = shift_mask[j].sum().item()
                total_lp = token_log_probs[j].sum().item()
                all_scores[idx] = total_lp / max(n_tokens, 1)

        return all_scores

    @torch.no_grad()
    def score_incremental(self, prefix: str, continuations: List[str]) -> List[float]:
        """
        Score multiple continuations given a shared prefix.

        **True incremental scoring**: computes prefix KV states once,
        then for each continuation scores only the continuation tokens
        conditioned on the prefix, with proper length normalisation
        on continuation tokens only.

        Returns:
            score = (1 / n_cont_tokens) * sum_t log P(cont_t | prefix, cont_{<t})

        This avoids:
          - Re-scoring the prefix for every candidate
          - Double-counting prefix likelihood
          - Length bias toward shorter continuations

        Parameters
        ----------
        prefix : str
            Fixed prefix text.
        continuations : list of str
            Candidate continuations to score.

        Returns
        -------
        list of float
            Length-normalised log-prob of continuation tokens only.
        """
        self._ensure_loaded()

        if not continuations:
            return []

        device = self._get_device()

        # Tokenise prefix to determine its token length
        if prefix.strip():
            prefix_inputs = self._tokenizer(
                prefix,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            prefix_ids = prefix_inputs["input_ids"].to(device)
            prefix_len = prefix_ids.shape[1]

            # Forward pass on prefix to cache KV states
            prefix_out = self._model(prefix_ids, use_cache=True)
            cached_kv = prefix_out.past_key_values
        else:
            prefix_ids = None
            prefix_len = 0
            cached_kv = None

        results: List[float] = []
        for cont in continuations:
            if not cont.strip():
                results.append(0.0)
                continue

            # Tokenise the full string to get correct boundary tokenisation
            full_text = (prefix + " " + cont).strip() if prefix.strip() else cont
            full_inputs = self._tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            full_ids = full_inputs["input_ids"].to(device)
            full_len = full_ids.shape[1]

            if full_len <= prefix_len or full_len <= 1:
                results.append(0.0)
                continue

            n_cont = full_len - prefix_len
            if n_cont <= 0:
                results.append(0.0)
                continue

            if cached_kv is not None and prefix_len > 0:
                # Use cached KV for efficiency, but do a single forward
                # pass on continuation tokens
                cont_token_ids = full_ids[:, prefix_len:]
                cont_out = self._model(
                    cont_token_ids,
                    past_key_values=cached_kv,
                    use_cache=False,
                )
                # cont_out.logits: [1, n_cont, vocab]
                # cont_out.logits[:, t, :] predicts token at position prefix_len + t + 1
                # But the first cont token (at prefix_len) is predicted by
                # prefix_out.logits[:, -1, :] (the last prefix logit)

                # Assemble scoring logits:
                # - Position 0: prefix's last logit predicts cont_token[0]
                # - Position 1..n_cont-1: cont_out.logits[:, 0..n_cont-2, :] predict cont_token[1..n_cont-1]
                first_logit = prefix_out.logits[:, -1:, :]  # [1, 1, V]
                if n_cont > 1:
                    rest_logits = cont_out.logits[:, :-1, :]  # [1, n_cont-1, V]
                    scoring_logits = torch.cat(
                        [first_logit, rest_logits], dim=1
                    )  # [1, n_cont, V]
                else:
                    scoring_logits = first_logit  # [1, 1, V]

                cont_labels = full_ids[:, prefix_len : prefix_len + n_cont]

                log_probs = torch.nn.functional.log_softmax(scoring_logits, dim=-1)
                token_lps = log_probs.gather(2, cont_labels.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, n_cont]

                total_lp = token_lps.sum().item()
                results.append(total_lp / max(n_cont, 1))
            else:
                # No prefix or empty prefix — full forward pass, score
                # continuation portion only
                if full_len <= 1:
                    results.append(0.0)
                    continue
                full_out = self._model(full_ids)
                logits = full_out.logits
                # Score continuation tokens only
                start = max(prefix_len - 1, 0)
                end = full_len - 1
                scoring_logits = logits[:, start:end, :]
                scoring_labels = full_ids[:, start + 1 : end + 1]

                if scoring_logits.shape[1] == 0:
                    results.append(0.0)
                    continue

                log_probs = torch.nn.functional.log_softmax(scoring_logits, dim=-1)
                token_lps = log_probs.gather(2, scoring_labels.unsqueeze(-1)).squeeze(
                    -1
                )

                n_scored = token_lps.shape[1]
                total_lp = token_lps.sum().item()
                results.append(total_lp / max(n_scored, 1))

        return results
