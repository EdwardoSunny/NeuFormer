"""
N-gram WFST Decoder
====================
Wrapper around the C++ ``lm_decoder`` pybind11 module from the NEJM
brain-to-text repo.  This provides Kaldi-based WFST CTC beam search
with n-gram language model integration.

The C++ module performs CTC-WFST beam search using a pre-compiled
TLG.fst (Token + Lexicon + Grammar) and optionally rescores the
resulting lattice with an unpruned G.fst.

Usage
-----
    from neural_decoder.ngram_decoder import NgramWFSTDecoder
    decoder = NgramWFSTDecoder(lm_path="/path/to/lm_dir")
    results = decoder.decode(logits, blank_penalty=9.0)
    # results is a list of dicts: {sentence, ac_score, lm_score}
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NgramDecodeResult:
    """Single hypothesis from WFST n-gram decoding."""

    sentence: str
    ac_score: float
    lm_score: float


class NgramWFSTDecoder:
    """
    Wrapper around the ``lm_decoder`` C++ pybind11 module.

    The module must be built from the NEJM brain-to-text repo's
    ``language_model/runtime/server/x86/`` directory.  See build
    instructions at the bottom of this file.

    Parameters
    ----------
    lm_path : str
        Path to directory containing ``TLG.fst`` and ``words.txt``.
        Optionally also ``G.fst`` and ``G_no_prune.fst`` for rescoring.
    max_active : int
        Maximum active states in WFST beam search.
    min_active : int
        Minimum active states.
    beam : float
        Beam width for decoding.
    lattice_beam : float
        Beam width for lattice generation.
    acoustic_scale : float
        Scale applied to acoustic (CTC) scores.
    ctc_blank_skip_threshold : float
        Skip frames where blank prob exceeds this threshold.
    length_penalty : float
        Penalty per token to control output length.
    nbest : int
        Number of n-best hypotheses to return.
    """

    def __init__(
        self,
        lm_path: str,
        max_active: int = 7000,
        min_active: int = 200,
        beam: float = 17.0,
        lattice_beam: float = 8.0,
        acoustic_scale: float = 0.3,
        ctc_blank_skip_threshold: float = 1.0,
        length_penalty: float = 0.0,
        nbest: int = 100,
    ):
        self.lm_path = os.path.expanduser(lm_path)
        self.nbest = nbest
        self.acoustic_scale = acoustic_scale

        if not os.path.exists(self.lm_path):
            raise ValueError(f"Language model path does not exist: {self.lm_path}")

        try:
            import lm_decoder

            self._lm_decoder = lm_decoder
        except ImportError:
            raise ImportError(
                "The lm_decoder C++ module is not installed. "
                "Build it from the language_model/runtime/server/x86/ directory. "
                "See build instructions in this file's docstring."
            )

        TLG_path = os.path.join(self.lm_path, "TLG.fst")
        words_path = os.path.join(self.lm_path, "words.txt")
        G_path = os.path.join(self.lm_path, "G.fst")
        rescore_G_path = os.path.join(self.lm_path, "G_no_prune.fst")

        if not os.path.exists(rescore_G_path):
            rescore_G_path = ""
            G_path = ""
        if not os.path.exists(G_path):
            G_path = ""
        if not os.path.exists(TLG_path):
            raise ValueError(f"TLG.fst not found at {TLG_path}")
        if not os.path.exists(words_path):
            raise ValueError(f"words.txt not found at {words_path}")

        decode_opts = lm_decoder.DecodeOptions(
            max_active,
            min_active,
            beam,
            lattice_beam,
            acoustic_scale,
            ctc_blank_skip_threshold,
            length_penalty,
            nbest,
        )

        decode_resource = lm_decoder.DecodeResource(
            TLG_path,
            G_path,
            rescore_G_path,
            words_path,
            "",  # unit_path (not used for speech)
        )

        self._decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)
        self._has_rescore = rescore_G_path != ""

        logger.info(f"WFST decoder initialized from {self.lm_path}")
        logger.info(f"  Rescoring available: {self._has_rescore}")

    def update_params(
        self,
        max_active: int = 7000,
        min_active: int = 200,
        beam: float = 17.0,
        lattice_beam: float = 8.0,
        acoustic_scale: float = 0.3,
        ctc_blank_skip_threshold: float = 1.0,
        length_penalty: float = 0.0,
        nbest: int = 100,
    ):
        """Update decoder parameters without re-loading FSTs."""
        self.acoustic_scale = acoustic_scale
        self.nbest = nbest

        decode_opts = self._lm_decoder.DecodeOptions(
            max_active,
            min_active,
            beam,
            lattice_beam,
            acoustic_scale,
            ctc_blank_skip_threshold,
            length_penalty,
            nbest,
        )
        self._decoder.SetOpt(decode_opts)

    def reset(self):
        """Reset decoder state (call between utterances)."""
        self._decoder.Reset()

    def decode(
        self,
        logits: np.ndarray,
        blank_penalty: float = 9.0,
        rescore: bool = True,
    ) -> List[NgramDecodeResult]:
        """
        Decode a single utterance.

        Parameters
        ----------
        logits : np.ndarray, shape [T, 41]
            Logits or log-probs from the Conformer.
            Expected order: [BLANK, SIL, phonemes...] (Kaldi order).
            Use ``rearrange_logits()`` to convert from model order.

            NOTE: The Conformer outputs log_softmax (log-probs), but
            ``DecodeNumpy`` applies log_softmax again internally. For
            peaked distributions (typical of trained CTC models) this
            still works well in practice. If this becomes an issue,
            use ``DecodeNumpyLogProbs`` (if available in your build)
            which skips the internal softmax.
        blank_penalty : float
            Penalty applied to blank emissions (as log(penalty)).
            Higher values encourage more non-blank emissions.
            Paper default: 9.0 for offline, 7.0 for online.
        rescore : bool
            Whether to rescore with unpruned G.fst if available.

        Returns
        -------
        List[NgramDecodeResult]
            N-best hypotheses sorted by score.
        """
        self._decoder.Reset()

        # lm_decoder.DecodeNumpy expects:
        #   logits: [T, 41] float32
        #   log_priors: [T, 41] float32 (zeros = no prior subtraction)
        #   blank_penalty: float (log of the penalty value)
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        log_priors = np.zeros_like(logits)

        self._lm_decoder.DecodeNumpy(
            self._decoder,
            logits,
            log_priors,
            np.log(blank_penalty),
        )

        self._decoder.FinishDecoding()

        # Optionally rescore with unpruned LM
        if rescore and self._has_rescore:
            self._decoder.Rescore()

        # Extract results
        results = []
        for r in self._decoder.result():
            results.append(
                NgramDecodeResult(
                    sentence=r.sentence.strip(),
                    ac_score=r.ac_score,
                    lm_score=r.lm_score,
                )
            )

        return results

    def decode_streaming(
        self,
        logits: np.ndarray,
        blank_penalty: float = 9.0,
    ) -> str:
        """
        Feed logits incrementally (for streaming/online use).
        Does NOT reset or finalize — call reset() before and
        finish_and_get_results() after.

        Parameters
        ----------
        logits : np.ndarray, shape [T, 41]
            A chunk of logits.
        blank_penalty : float
            Blank penalty.

        Returns
        -------
        str
            Current partial decoded sentence.
        """
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        log_priors = np.zeros_like(logits)

        self._lm_decoder.DecodeNumpy(
            self._decoder,
            logits,
            log_priors,
            np.log(blank_penalty),
        )

        if self._decoder.DecodedSomething():
            return self._decoder.result()[0].sentence.strip()
        return ""

    def finish_and_get_results(
        self,
        rescore: bool = True,
    ) -> List[NgramDecodeResult]:
        """
        Finalize streaming decoding and return results.
        Call after all chunks have been fed via decode_streaming().
        """
        self._decoder.FinishDecoding()

        if rescore and self._has_rescore:
            self._decoder.Rescore()

        results = []
        for r in self._decoder.result():
            results.append(
                NgramDecodeResult(
                    sentence=r.sentence.strip(),
                    ac_score=r.ac_score,
                    lm_score=r.lm_score,
                )
            )
        return results


def rearrange_logits(logits: np.ndarray) -> np.ndarray:
    """
    Rearrange logits from model order to Kaldi/WFST order.

    Model order:  [BLANK, phonemes(1-39), SIL(40)]
    Kaldi order:  [BLANK, SIL, phonemes(1-39)]

    Parameters
    ----------
    logits : np.ndarray, shape [..., 41]
        Logits in model order (last dim is classes).

    Returns
    -------
    np.ndarray
        Logits rearranged to Kaldi order.
    """
    # Move SIL (index 40) to position 1, shift phonemes right
    return np.concatenate(
        [logits[..., 0:1], logits[..., -1:], logits[..., 1:-1]],
        axis=-1,
    )
