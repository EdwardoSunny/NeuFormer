"""
CTC Beam Search Decoder with KenLM n-gram Language Model
==========================================================
Uses ``torchaudio.models.decoder.ctc_decoder`` for phoneme-level CTC beam
search decoding with lexicon constraint and KenLM language model support.

Based on the torchaudio ASR inference tutorial:
https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html

The Conformer outputs CTC log-probs over 41 classes::

    [BLANK(0), AA(1), AE(2), ..., ZH(39), SIL(40)]

Components
----------
- **Tokens**: the 41 CTC output symbols (``-`` for blank, 39 ARPABET phones,
  ``|`` for silence/word-boundary).
- **Lexicon**: mapping from words to phoneme sequences, constraining beam
  search to only produce valid words.
- **Language Model**: KenLM n-gram (``.arpa`` or ``.bin``), or a custom LM
  that inherits ``CTCDecoderLM``.

Required files in ``lm_dir``::

    lexicon.txt      word → phoneme mapping  (auto-generated from
                     lexicon_numbers.txt if missing)
    tokens.txt       one token per line      (auto-generated if missing)
    lm.arpa or lm.bin   KenLM language model (optional — omit for no LM)

Usage
-----
::

    decoder = CTCBeamSearchDecoder("path/to/lm_dir")
    text    = decoder.decode(log_probs)           # best hypothesis
    nbest   = decoder.decode_nbest(log_probs, 5)  # top-5

Custom LM
----------
You can also pass a custom ``torch.nn.Module`` language model via the
``CTCDecoderLM`` / ``CTCDecoderLMState`` API::

    from neural_decoder.ngram_decoder import CTCBeamSearchDecoder
    from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState

    class MyLM(CTCDecoderLM):
        ...

    decoder = CTCBeamSearchDecoder("path/to/lm_dir", lm=MyLM(my_model))
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phoneme table (must match Conformer CTC output indices)
# ---------------------------------------------------------------------------
PHONE_DEF = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
BLANK_TOKEN = "-"
SIL_TOKEN = "|"
TOKENS: List[str] = [BLANK_TOKEN] + PHONE_DEF + [SIL_TOKEN]  # len == 41


@dataclass
class Hypothesis:
    """A single decoded hypothesis."""

    words: List[str]
    score: float
    tokens: List[int]
    timesteps: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Greedy CTC decoder (no LM)
# ---------------------------------------------------------------------------
class GreedyCTCDecoder(torch.nn.Module):
    """Greedy (argmax) CTC decoder — no language model, no beam search.

    Given a sequence of CTC emissions, takes the argmax at each timestep,
    collapses consecutive duplicates, and removes blanks.

    Parameters
    ----------
    tokens : list of str
        The 41 CTC output tokens in model order.
    blank : int
        Index of the CTC blank token (default 0).
    """

    def __init__(self, tokens: List[str] = TOKENS, blank: int = 0):
        super().__init__()
        self.tokens = tokens
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> list:
        """Decode a single utterance.

        Parameters
        ----------
        emission : Tensor
            Shape ``[T, n_classes]`` — log-probs or logits.

        Returns
        -------
        list of str
            Decoded words (phonemes joined, ``|`` treated as word boundary).
        """
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.tokens[i] for i in indices])
        return joined.replace("|", " ").strip().split()


# ---------------------------------------------------------------------------
# Beam search CTC decoder with KenLM
# ---------------------------------------------------------------------------
class CTCBeamSearchDecoder:
    """CTC beam search decoder with optional KenLM n-gram language model.

    Wraps ``torchaudio.models.decoder.ctc_decoder`` following the official
    torchaudio tutorial pattern.

    Parameters
    ----------
    lm_dir : str
        Directory containing ``lexicon.txt``, ``tokens.txt``, and optionally
        ``lm.arpa`` / ``lm.bin``.
    lm_weight : float
        Language model weight (higher → more LM influence).
    word_score : float
        Per-word insertion bonus (negative = penalty).
    sil_score : float
        Per-silence appearance bonus.
    beam_size : int
        Maximum number of active beam hypotheses.
    beam_size_token : int or None
        Number of top tokens to consider per step. ``None`` uses all tokens.
    beam_threshold : float
        Pruning threshold — hypotheses with scores more than this below the
        best are discarded.
    nbest : int
        Maximum number of n-best hypotheses to return.
    blank_penalty : float
        Value *subtracted* from blank log-prob (log-space) to discourage
        blank emissions.
    lm : optional
        A custom LM object (e.g. a ``CTCDecoderLM`` subclass wrapping a
        neural LM). If provided, overrides the KenLM file in ``lm_dir``.
    """

    def __init__(
        self,
        lm_dir: str,
        lm_weight: float = 3.23,
        word_score: float = -0.26,
        sil_score: float = 0.0,
        beam_size: int = 1500,
        beam_size_token: Optional[int] = None,
        beam_threshold: float = 50.0,
        nbest: int = 1,
        blank_penalty: float = 0.0,
        lm: Optional[object] = None,
    ):
        self.lm_dir = os.path.expanduser(lm_dir)
        self.blank_penalty = blank_penalty

        # ---- ensure required files exist --------------------------------
        lex_path = os.path.join(self.lm_dir, "lexicon.txt")
        tok_path = os.path.join(self.lm_dir, "tokens.txt")

        if not os.path.exists(lex_path):
            num_lex = os.path.join(self.lm_dir, "lexicon_numbers.txt")
            if os.path.exists(num_lex):
                _convert_numeric_lexicon(num_lex, lex_path)
            else:
                raise FileNotFoundError(
                    f"Need lexicon.txt or lexicon_numbers.txt in {self.lm_dir}"
                )
        if not os.path.exists(tok_path):
            _write_tokens_file(tok_path)

        # ---- resolve language model -------------------------------------
        if lm is not None:
            lm_source = lm  # custom CTCDecoderLM
        else:
            lm_source = _find_lm(self.lm_dir)  # path to .arpa/.bin or None

        # ---- build decoder via torchaudio factory -----------------------
        from torchaudio.models.decoder import ctc_decoder

        decoder_kwargs = dict(
            lexicon=lex_path,
            tokens=tok_path,
            lm=lm_source,
            nbest=nbest,
            beam_size=beam_size,
            beam_threshold=beam_threshold,
            lm_weight=lm_weight,
            word_score=word_score,
            sil_score=sil_score,
            blank_token=BLANK_TOKEN,
            sil_token=SIL_TOKEN,
        )
        if beam_size_token is not None:
            decoder_kwargs["beam_size_token"] = beam_size_token

        self._decoder = ctc_decoder(**decoder_kwargs)

        logger.info(
            "CTCBeamSearchDecoder ready  lm=%s  beam=%d  lm_weight=%.2f",
            "custom" if lm is not None else ("yes" if lm_source else "none"),
            beam_size,
            lm_weight,
        )

    # ------------------------------------------------------------------
    # Batch decoding
    # ------------------------------------------------------------------
    def __call__(
        self,
        emission: torch.Tensor,
    ) -> list:
        """Decode a batch of emissions (mirrors torchaudio API).

        Parameters
        ----------
        emission : Tensor
            Shape ``[B, T, C]`` — log-probs or logits.

        Returns
        -------
        list of list of CTCHypothesis
            Outer list is per-batch, inner list is per-nbest.
        """
        return self._decoder(emission)

    # ------------------------------------------------------------------
    # Convenience: single-utterance decode
    # ------------------------------------------------------------------
    def decode(
        self,
        emission: np.ndarray,
        is_log_probs: bool = True,
    ) -> List[str]:
        """Return the best hypothesis words for one utterance.

        Parameters
        ----------
        emission : ndarray
            Shape ``[T, C]`` (C=41 for this model).
        is_log_probs : bool
            True if already log-softmax'd (Conformer default).

        Returns
        -------
        list of str
            Words from the best hypothesis.
        """
        results = self.decode_nbest(emission, n=1, is_log_probs=is_log_probs)
        return results[0].words if results else []

    def decode_nbest(
        self,
        emission: np.ndarray,
        n: int = 5,
        is_log_probs: bool = True,
    ) -> List[Hypothesis]:
        """Return top-*n* hypotheses for one utterance.

        Parameters
        ----------
        emission : ndarray
            Shape ``[T, C]``.
        n : int
            Number of hypotheses to return.
        is_log_probs : bool
            Whether the input is already log-softmax'd.

        Returns
        -------
        list of Hypothesis
        """
        em = torch.from_numpy(np.ascontiguousarray(emission, dtype=np.float32))

        if not is_log_probs:
            em = torch.nn.functional.log_softmax(em, dim=-1)

        if self.blank_penalty > 0:
            em[:, 0] -= self.blank_penalty

        # ctc_decoder expects [B, T, C]
        out = self._decoder(em.unsqueeze(0))
        hypotheses = []
        for hyp in out[0][:n]:
            hypotheses.append(
                Hypothesis(
                    words=list(hyp.words) if hyp.words else [],
                    score=hyp.score,
                    tokens=hyp.tokens.tolist()
                    if hasattr(hyp.tokens, "tolist")
                    else list(hyp.tokens),
                    timesteps=hyp.timesteps if hasattr(hyp, "timesteps") else None,
                )
            )
        return hypotheses

    def decode_text(
        self,
        emission: np.ndarray,
        is_log_probs: bool = True,
    ) -> str:
        """Return the best hypothesis as a plain text string.

        Parameters
        ----------
        emission : ndarray
            Shape ``[T, C]``.
        is_log_probs : bool
            Whether the input is already log-softmax'd.
        """
        words = self.decode(emission, is_log_probs=is_log_probs)
        return " ".join(words).strip() if words else ""

    # ------------------------------------------------------------------
    # Incremental (streaming) decoding
    # ------------------------------------------------------------------
    def decode_begin(self):
        """Initialize internal state for incremental decoding."""
        self._decoder.decode_begin()

    def decode_step(self, emission_frame: torch.Tensor):
        """Feed one frame (or a few frames) to the incremental decoder.

        Parameters
        ----------
        emission_frame : Tensor
            Shape ``[1, C]`` or ``[F, C]`` for F frames.
        """
        self._decoder.decode_step(emission_frame)

    def decode_end(self):
        """Finalize incremental decoding."""
        self._decoder.decode_end()

    def get_final_hypothesis(self) -> list:
        """Retrieve hypotheses after incremental decoding."""
        return self._decoder.get_final_hypothesis()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_lm(lm_dir: str) -> Optional[str]:
    """Search for a KenLM file in *lm_dir*."""
    for name in ("lm.bin", "lm.arpa", "lm_orig.arpa", "lm_pruned.arpa"):
        p = os.path.join(lm_dir, name)
        if os.path.exists(p):
            logger.info("Found LM: %s", p)
            return p
    logger.warning("No KenLM file found in %s — decoding without LM", lm_dir)
    return None


def _convert_numeric_lexicon(src: str, dst: str):
    """Convert ``lexicon_numbers.txt`` → torchaudio ``lexicon.txt``.

    ``lexicon_numbers.txt`` format::

        WORD id1 id2 ...

    where token IDs use model order: 1=AA … 39=ZH, 40=SIL.

    torchaudio ``lexicon.txt`` format::

        word PHONEME PHONEME ... |

    Words are lowercased to match standard KenLM language models
    (e.g. LibriSpeech 4-gram) which are trained on lowercase text.
    """
    id2ph = {i + 1: p for i, p in enumerate(PHONE_DEF)}
    id2ph[len(PHONE_DEF) + 1] = SIL_TOKEN  # 40 → |

    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            phones = [id2ph.get(int(t), "") for t in parts[1:]]
            phones = [p for p in phones if p]  # drop unknowns
            if not phones or phones[-1] != SIL_TOKEN:
                phones.append(SIL_TOKEN)
            fout.write(f"{word} {' '.join(phones)}\n")
    logger.info("Wrote %s", dst)


def _write_tokens_file(path: str):
    """Write the default 41-token file (blank + 39 phones + SIL)."""
    with open(path, "w") as f:
        for tok in TOKENS:
            f.write(tok + "\n")


# ---------------------------------------------------------------------------
# GPT-2 Neural Language Model for CTC beam search
# ---------------------------------------------------------------------------


class GPT2CTCDecoderLM:
    """GPT-2 language model adapter for ``torchaudio.models.decoder.ctc_decoder``.

    Wraps a HuggingFace GPT-2 model as a ``CTCDecoderLM`` so it can be used
    in lexicon-constrained CTC beam search.  At each word boundary the LM
    scores the new word in context using the GPT-2 cross-entropy, providing
    much stronger rescoring than KenLM n-grams.

    Based on B2T_Model's ``CustomLM`` implementation.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID (default ``"gpt2"``).
    device : str
        Device for the GPT-2 model (default ``"cuda"``).

    Usage
    -----
    ::

        from neural_decoder.ngram_decoder import GPT2CTCDecoderLM, CTCBeamSearchDecoder

        gpt2_lm = GPT2CTCDecoderLM("gpt2", device="cuda")
        decoder = CTCBeamSearchDecoder(
            lm_dir="lm_dir",
            lm=gpt2_lm.build_lm("lm_dir/lexicon.txt", "lm_dir/tokens.txt"),
            lm_weight=2.0,
            beam_size=50,
        )
    """

    def __init__(self, model_id: str = "gpt2", device: str = "cuda"):
        from transformers import AutoTokenizer, GPT2LMHeadModel

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = GPT2LMHeadModel.from_pretrained(model_id).eval().to(device)

    def build_lm(self, lexicon_path: str, tokens_path: str):
        """Build and return a ``CTCDecoderLM`` that wraps this GPT-2 model.

        The returned object can be passed as the ``lm`` argument to
        ``CTCBeamSearchDecoder``.

        Parameters
        ----------
        lexicon_path : str
            Path to ``lexicon.txt``.
        tokens_path : str
            Path to ``tokens.txt``.
        """
        from flashlight.lib.text.decoder import (
            LM as _LM,
            LMState as _LMState,
        )
        from flashlight.lib.text.dictionary import (
            create_word_dict as _create_word_dict,
            load_words as _load_words,
        )

        lexicon = _load_words(lexicon_path)
        self._word_dict = _create_word_dict(lexicon)

        parent = self

        class _GPT2LM(_LM):
            """Flashlight LM interface backed by GPT-2."""

            def __init__(self):
                _LM.__init__(self)
                self.infos = {}

            def _past_kv_to_device(self, pkv, device):
                return tuple(tuple(t.to(device) for t in layer) for layer in pkv)

            def start(self, start_with_nothing: bool) -> _LMState:
                state = _LMState()
                tokenized = parent.tokenizer.encode(
                    parent.tokenizer.bos_token, return_tensors="pt"
                )[0].to(parent.device)
                with torch.no_grad():
                    output = parent.model(tokenized, past_key_values=None)
                self.infos[state] = {
                    "score": 0.0,
                    "last_token_logit": output.logits[-1].cpu(),
                    "past_key_values": self._past_kv_to_device(
                        output.past_key_values, "cpu"
                    ),
                }
                return state

            def score(self, state: _LMState, usr_token_idx: int):
                word = parent._word_dict.get_entry(usr_token_idx)
                return self._score_word(state, word, usr_token_idx)

            def _score_word(self, state, word, usr_token_idx):
                new_state = state.child(usr_token_idx)
                if new_state in self.infos:
                    return new_state, self.infos[new_state]["score"]

                info = self.infos[state]
                input_ids = parent.tokenizer.encode(" " + word, return_tensors="pt")[
                    0
                ].to(parent.device)
                last_logit = info["last_token_logit"].to(parent.device)
                pkv = self._past_kv_to_device(info["past_key_values"], parent.device)

                with torch.no_grad():
                    output = parent.model(input_ids, past_key_values=pkv)

                # Score = 1 / (cross-entropy + eps)
                lm_logits = torch.cat(
                    [last_logit.unsqueeze(0), output.logits[:-1]], dim=0
                )
                loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
                score = 1.0 / (loss_fn(lm_logits, input_ids) + 0.001)

                self.infos[new_state] = {
                    "score": score.cpu().item(),
                    "last_token_logit": output.logits[-1].cpu(),
                    "past_key_values": self._past_kv_to_device(
                        output.past_key_values, "cpu"
                    ),
                }
                return new_state, score.cpu().item()

            def finish(self, state: _LMState):
                return self._score_word(state, parent.tokenizer.eos_token, -1)

        return _GPT2LM()
