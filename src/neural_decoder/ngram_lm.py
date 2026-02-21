"""
N-gram Language Model for Shallow Fusion
=========================================
Provides word-level N-gram scoring for integration during beam search
and first-pass rescoring, following the approach in Willett et al. (2023).

Three tiers of N-gram integration:
  1. **During lexicon segmentation**: score word candidates as they're
     selected, biasing toward linguistically plausible word sequences.
  2. **First-pass rescoring**: N-gram scores the full N-best list cheaply.
  3. **Combined with LLM**: final score = alpha*CTC + beta*ngram + (1-beta)*LLM.

Supports two backends:
  - **KenLM** (preferred): fast C++ N-gram via the `kenlm` package.
  - **Pure Python** (fallback): simple interpolated N-gram trained from
    the training sentences.  Slower but no external dependencies.

Usage
-----
    from neural_decoder.ngram_lm import NgramLM
    lm = NgramLM(order=5)
    lm.train(["the cat sat on the mat", "hello world"])
    score = lm.score_sentence("the cat sat")
    word_score = lm.score_word("sat", context=["the", "cat"])
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


class NgramLM:
    """
    N-gram language model with Kneser-Ney-like smoothing.

    Parameters
    ----------
    order : int
        N-gram order (3 = trigram, 5 = 5-gram).
    kenlm_path : str or None
        Path to a pre-built KenLM ARPA/binary model.  If provided,
        uses KenLM backend (fast).  Otherwise trains a pure-Python
        model from sentences.
    discount : float
        Absolute discount for modified Kneser-Ney smoothing.
    """

    def __init__(
        self,
        order: int = 5,
        kenlm_path: Optional[str] = None,
        discount: float = 0.75,
    ):
        self.order = order
        self.discount = discount
        self._kenlm_model = None
        self._kenlm_path = kenlm_path

        # Pure-Python fallback state
        self._ngram_counts: Dict[int, Dict[Tuple[str, ...], int]] = {}
        self._context_counts: Dict[int, Dict[Tuple[str, ...], int]] = {}
        self._vocab: set = set()
        self._total_unigrams: int = 0
        self._trained = False

        if kenlm_path:
            self._load_kenlm(kenlm_path)

    def _load_kenlm(self, path: str):
        """Load a KenLM model."""
        try:
            import kenlm

            self._kenlm_model = kenlm.Model(path)
        except ImportError:
            print(
                "Warning: kenlm not installed. "
                "Install with: pip install https://github.com/kpu/kenlm/archive/master.zip"
            )
            self._kenlm_model = None
        except Exception as e:
            print(f"Warning: failed to load KenLM model: {e}")
            self._kenlm_model = None

    def train(self, sentences: List[str]):
        """
        Train the N-gram model from a list of sentences.

        If KenLM is available, builds a KenLM model via lmplz.
        Otherwise, trains a pure-Python model with absolute discounting.
        """
        if not sentences:
            return

        # Try KenLM first
        if self._try_train_kenlm(sentences):
            self._trained = True
            return

        # Pure-Python fallback
        self._train_python(sentences)
        self._trained = True

    def _try_train_kenlm(self, sentences: List[str]) -> bool:
        """Try to train via KenLM's lmplz (much faster, better smoothing)."""
        try:
            import kenlm
            import subprocess

            # Write sentences to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                for s in sentences:
                    cleaned = s.strip().lower()
                    if cleaned:
                        f.write(cleaned + "\n")
                text_path = f.name

            arpa_path = text_path + ".arpa"
            bin_path = text_path + ".binary"

            # Run lmplz
            try:
                result = subprocess.run(
                    [
                        "lmplz",
                        "-o",
                        str(self.order),
                        "--text",
                        text_path,
                        "--arpa",
                        arpa_path,
                        "--discount_fallback",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"lmplz failed: {result.stderr}")
            except FileNotFoundError:
                # lmplz not on PATH — try to build with build_binary
                os.unlink(text_path)
                return False

            # Convert to binary for faster loading
            try:
                subprocess.run(
                    ["build_binary", arpa_path, bin_path],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                model_path = bin_path if os.path.exists(bin_path) else arpa_path
            except FileNotFoundError:
                model_path = arpa_path

            self._kenlm_model = kenlm.Model(model_path)
            self._kenlm_path = model_path

            # Cleanup temp files
            try:
                os.unlink(text_path)
                if os.path.exists(arpa_path) and model_path != arpa_path:
                    os.unlink(arpa_path)
            except OSError:
                pass

            return True

        except (ImportError, Exception):
            return False

    def _train_python(self, sentences: List[str]):
        """Train pure-Python N-gram with absolute discounting."""
        self._ngram_counts = {n: Counter() for n in range(1, self.order + 1)}
        self._context_counts = {n: Counter() for n in range(1, self.order + 1)}
        self._vocab = set()

        for sent in sentences:
            words = ["<s>"] + sent.strip().lower().split() + ["</s>"]
            self._vocab.update(words)

            for n in range(1, self.order + 1):
                for i in range(len(words) - n + 1):
                    ngram = tuple(words[i : i + n])
                    self._ngram_counts[n][ngram] += 1
                    if n > 1:
                        context = ngram[:-1]
                        self._context_counts[n][context] += 1

        self._total_unigrams = sum(self._ngram_counts[1].values())
        self._vocab.add("<unk>")

    @property
    def is_ready(self) -> bool:
        return self._kenlm_model is not None or self._trained

    def score_sentence(self, sentence: str) -> float:
        """
        Score a full sentence.  Returns length-normalized log10 probability.

        For KenLM: uses KenLM's native scoring.
        For Python: sums per-word log-probs with backoff.
        """
        if not self.is_ready:
            return 0.0

        if self._kenlm_model is not None:
            # KenLM returns total log10 prob
            total = self._kenlm_model.score(sentence.strip().lower())
            n_words = len(sentence.strip().split())
            return total / max(n_words, 1)

        # Pure Python
        words = sentence.strip().lower().split()
        if not words:
            return 0.0

        total_logp = 0.0
        context = ["<s>"]
        for word in words:
            logp = self._score_word_python(word, context)
            total_logp += logp
            context.append(word)
            if len(context) >= self.order:
                context = context[-(self.order - 1) :]

        # Add </s>
        total_logp += self._score_word_python("</s>", context)
        n_tokens = len(words) + 1  # +1 for </s>

        return total_logp / max(n_tokens, 1)

    def score_sentence_raw(self, sentence: str) -> float:
        """Score a sentence, returning the total (non-normalized) log10 prob."""
        if not self.is_ready:
            return 0.0

        if self._kenlm_model is not None:
            return self._kenlm_model.score(sentence.strip().lower())

        words = sentence.strip().lower().split()
        if not words:
            return 0.0

        total_logp = 0.0
        context = ["<s>"]
        for word in words:
            total_logp += self._score_word_python(word, context)
            context.append(word)
            if len(context) >= self.order:
                context = context[-(self.order - 1) :]
        total_logp += self._score_word_python("</s>", context)
        return total_logp

    def score_word(self, word: str, context: List[str]) -> float:
        """
        Score a single word given preceding context.

        Returns log10 P(word | context).

        Parameters
        ----------
        word : str
        context : list of str
            Preceding words (most recent last).
        """
        if not self.is_ready:
            return 0.0

        word = word.lower()

        if self._kenlm_model is not None:
            return self._score_word_kenlm(word, context)

        return self._score_word_python(word, context)

    def _score_word_kenlm(self, word: str, context: List[str]) -> float:
        """Score a word using KenLM's full_scores."""
        # Build the context + word string
        ctx = " ".join(c.lower() for c in context[-self.order + 1 :])
        full = (ctx + " " + word).strip()

        # KenLM's full_scores returns per-word scores
        # We use score() on the full sequence minus score of context
        score_full = self._kenlm_model.score(full)
        if context:
            score_ctx = self._kenlm_model.score(ctx)
        else:
            score_ctx = 0.0

        return score_full - score_ctx

    def _score_word_python(self, word: str, context: List[str]) -> float:
        """
        Score a word using the Python N-gram with absolute discount backoff.

        Returns log10 P(word | context).
        """
        word = word.lower()
        if word not in self._vocab:
            word = "<unk>"

        # Try from highest order down to unigram
        for n in range(min(self.order, len(context) + 1), 0, -1):
            if n == 1:
                # Unigram
                count = self._ngram_counts[1].get((word,), 0)
                if count > 0:
                    return math.log10(count / max(self._total_unigrams, 1))
                else:
                    # Uniform over vocab
                    return math.log10(1.0 / max(len(self._vocab), 1))
            else:
                ctx = tuple(c.lower() for c in context[-(n - 1) :])
                ngram = ctx + (word,)
                ngram_count = self._ngram_counts[n].get(ngram, 0)
                ctx_count = self._context_counts[n].get(ctx, 0)

                if ctx_count > 0 and ngram_count > 0:
                    # Absolute discount
                    prob = max(ngram_count - self.discount, 0) / ctx_count
                    # Backoff weight
                    n_unique = len(
                        [
                            ng
                            for ng in self._ngram_counts[n]
                            if ng[:-1] == ctx and self._ngram_counts[n][ng] > 0
                        ]
                    )
                    backoff_weight = (self.discount * n_unique) / ctx_count
                    # Recursively get lower-order prob
                    lower_logp = self._score_word_python(word, list(ctx[1:]))
                    lower_prob = 10**lower_logp

                    total_prob = prob + backoff_weight * lower_prob
                    if total_prob > 0:
                        return math.log10(total_prob)

        # Final fallback
        return math.log10(1.0 / max(len(self._vocab), 1))

    def score_words_incremental(self, words: List[str]) -> List[float]:
        """
        Score each word in a sequence incrementally.

        Returns a list of per-word log10 P(w_i | w_{<i}).
        Useful for getting per-position N-gram scores during beam search.
        """
        if not words:
            return []

        scores = []
        context = ["<s>"]
        for word in words:
            s = self.score_word(word, context)
            scores.append(s)
            context.append(word)
            if len(context) >= self.order:
                context = context[-(self.order - 1) :]

        return scores

    def score_batch(self, sentences: List[str]) -> List[float]:
        """Score a batch of sentences. Returns normalized log10 probs."""
        return [self.score_sentence(s) for s in sentences]

    def score_batch_raw(self, sentences: List[str]) -> List[float]:
        """Score a batch of sentences. Returns raw (non-normalized) log10 probs."""
        return [self.score_sentence_raw(s) for s in sentences]
