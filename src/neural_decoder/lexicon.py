"""
A2 – Pronunciation Lexicon & Phoneme→Word Conversion
=====================================================
Builds a pronunciation lexicon from g2p_en and provides beam-search
segmentation to convert a phoneme sequence into a word sequence.

The lexicon maps word → list-of-phoneme-ID-sequences (multiple
pronunciations are possible).  Given a decoded phoneme string from
CTC beam search, we find the best word-level segmentation using
dynamic programming (Viterbi-style), optionally allowing small
edit-distance mismatches for robustness.

Usage
-----
    from neural_decoder.lexicon import PronunciationLexicon
    lex = PronunciationLexicon()
    lex.build_from_sentences(["hello world", "good morning"])
    words, score = lex.phonemes_to_words([17, 10, 20, 24, 40, ...])
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .phoneme_table import (
    BLANK_IDX,
    ID_TO_PHONE,
    PHONE_DEF_SIL,
    PHONE_TO_ID,
    SIL_IDX,
)


def _edit_distance(a: List[int], b: List[int]) -> int:
    """Standard Levenshtein edit distance between two integer sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


@dataclass
class LexiconEntry:
    """A word with one or more pronunciations."""

    word: str
    pronunciations: List[List[int]] = field(default_factory=list)


class PronunciationLexicon:
    """
    Pronunciation lexicon for phoneme → word conversion.

    Supports multiple pronunciation sources:
      1. CMUdict (base lexicon, loaded if available)
      2. g2p_en (for OOV words not in CMUdict)
      3. Grapheme spell-out (last-resort fallback)

    Parameters
    ----------
    max_edit_dist : int
        Maximum edit distance allowed when matching a phoneme substring
        to a lexicon entry.  0 = exact match only.
    sil_idx : int
        Index of the SIL (word boundary) token in the phoneme vocabulary.
    use_cmudict : bool
        If True, load CMUdict as a base pronunciation source.
    use_grapheme_fallback : bool
        If True, allow grapheme spell-out as a last-resort for OOV words.
    """

    def __init__(
        self,
        max_edit_dist: int = 1,
        sil_idx: int = SIL_IDX,
        use_cmudict: bool = True,
        use_grapheme_fallback: bool = True,
    ):
        self.max_edit_dist = max_edit_dist
        self.sil_idx = sil_idx
        self.use_cmudict = use_cmudict
        self.use_grapheme_fallback = use_grapheme_fallback
        self.entries: Dict[str, LexiconEntry] = {}
        # Reverse index: tuple of phoneme IDs → list of words
        self._pron_to_words: Dict[Tuple[int, ...], List[str]] = defaultdict(list)
        # Length-indexed reverse index for fast fuzzy matching:
        # pron_length → list of (word, pronunciation) pairs
        self._length_index: Dict[int, List[Tuple[str, List[int]]]] = defaultdict(list)
        self._g2p = None  # lazy loaded
        self._cmudict: Optional[Dict[str, List[List[str]]]] = None
        self._cmudict_loaded = False
        # OOV tracking
        self._oov_words: set = set()
        self._total_lookups: int = 0
        self._oov_lookups: int = 0

    # Grapheme-to-phoneme mapping for spell-out fallback
    _GRAPHEME_MAP: Dict[str, List[int]] = {}  # populated lazily

    def _get_g2p(self):
        if self._g2p is None:
            from g2p_en import G2p

            self._g2p = G2p()
        return self._g2p

    def _load_cmudict(self):
        """Load CMU Pronouncing Dictionary if available."""
        if self._cmudict_loaded:
            return
        self._cmudict_loaded = True
        try:
            import nltk

            try:
                self._cmudict = nltk.corpus.cmudict.dict()
            except LookupError:
                # Try to download it
                try:
                    nltk.download("cmudict", quiet=True)
                    self._cmudict = nltk.corpus.cmudict.dict()
                except Exception:
                    self._cmudict = None
        except ImportError:
            self._cmudict = None

    def _cmudict_pronunciations(self, word: str) -> List[List[int]]:
        """Get pronunciation(s) from CMUdict for a word."""
        if not self.use_cmudict:
            return []
        self._load_cmudict()
        if self._cmudict is None:
            return []

        word_lower = word.lower()
        if word_lower not in self._cmudict:
            return []

        results: List[List[int]] = []
        for pron in self._cmudict[word_lower]:
            ids = []
            for p in pron:
                p_clean = re.sub(r"[0-9]", "", p)
                if p_clean in PHONE_TO_ID:
                    ids.append(PHONE_TO_ID[p_clean])
            if ids:
                results.append(ids)
        return results

    def _grapheme_spell_out(self, word: str) -> List[int]:
        """
        Last-resort fallback: spell out the word as graphemes mapped
        to the closest phonemes.

        This allows handling of truly unknown words (e.g., proper nouns,
        technical terms) that neither CMUdict nor g2p can handle.
        """
        if not self.use_grapheme_fallback:
            return []

        # Simple mapping from common graphemes to phoneme IDs
        grapheme_to_phone = {
            "a": PHONE_TO_ID.get("AE", 2),
            "b": PHONE_TO_ID.get("B", 7),
            "c": PHONE_TO_ID.get("K", 20),
            "d": PHONE_TO_ID.get("D", 9),
            "e": PHONE_TO_ID.get("EH", 11),
            "f": PHONE_TO_ID.get("F", 14),
            "g": PHONE_TO_ID.get("G", 15),
            "h": PHONE_TO_ID.get("HH", 16),
            "i": PHONE_TO_ID.get("IH", 17),
            "j": PHONE_TO_ID.get("JH", 19),
            "k": PHONE_TO_ID.get("K", 20),
            "l": PHONE_TO_ID.get("L", 21),
            "m": PHONE_TO_ID.get("M", 22),
            "n": PHONE_TO_ID.get("N", 23),
            "o": PHONE_TO_ID.get("OW", 25),
            "p": PHONE_TO_ID.get("P", 27),
            "q": PHONE_TO_ID.get("K", 20),
            "r": PHONE_TO_ID.get("R", 28),
            "s": PHONE_TO_ID.get("S", 29),
            "t": PHONE_TO_ID.get("T", 31),
            "u": PHONE_TO_ID.get("AH", 3),
            "v": PHONE_TO_ID.get("V", 35),
            "w": PHONE_TO_ID.get("W", 36),
            "x": PHONE_TO_ID.get("K", 20),  # approximation
            "y": PHONE_TO_ID.get("Y", 37),
            "z": PHONE_TO_ID.get("Z", 38),
        }
        ids = []
        for ch in word.lower():
            if ch in grapheme_to_phone:
                ids.append(grapheme_to_phone[ch])
        return ids

    def _word_to_phoneme_ids(self, word: str) -> List[List[int]]:
        """
        Convert a word to phoneme IDs using the cascade:
          1. CMUdict (multiple pronunciations)
          2. g2p_en
          3. Grapheme spell-out (fallback)

        Returns a list of alternative pronunciations.
        """
        self._total_lookups += 1
        word_lower = word.lower()

        # 1. Try CMUdict first (highest quality)
        cmu_prons = self._cmudict_pronunciations(word_lower)
        if cmu_prons:
            return cmu_prons

        # 2. g2p_en
        try:
            g2p = self._get_g2p()
            raw = g2p(word_lower)
            ids = []
            for p in raw:
                p_clean = re.sub(r"[0-9]", "", p)
                if re.match(r"[A-Z]+", p_clean) and p_clean in PHONE_TO_ID:
                    ids.append(PHONE_TO_ID[p_clean])
            if ids:
                return [ids]
        except Exception:
            pass

        # 3. Grapheme spell-out (last resort)
        self._oov_lookups += 1
        self._oov_words.add(word_lower)
        spell_ids = self._grapheme_spell_out(word_lower)
        if spell_ids:
            return [spell_ids]

        return []

    def add_word(self, word: str, pronunciation: Optional[List[int]] = None):
        """Add a word (with optional explicit pronunciation) to the lexicon."""
        word_lower = word.lower()
        if pronunciation is not None:
            # Explicit pronunciation provided
            prons = [pronunciation]
        else:
            prons = self._word_to_phoneme_ids(word_lower)

        if not prons:
            return

        if word_lower not in self.entries:
            self.entries[word_lower] = LexiconEntry(word=word_lower)
        entry = self.entries[word_lower]

        for pron in prons:
            if not pron:
                continue
            if pron not in entry.pronunciations:
                entry.pronunciations.append(pron)
                self._pron_to_words[tuple(pron)].append(word_lower)
                self._length_index[len(pron)].append((word_lower, pron))

    def build_from_sentences(self, sentences: List[str]):
        """
        Build the lexicon from a list of training sentences.
        Each unique word is added with its pronunciation(s) from
        CMUdict/g2p/grapheme cascade.
        """
        seen: set = set()
        for sent in sentences:
            cleaned = re.sub(r"[^a-zA-Z\- ']", "", sent).lower()
            cleaned = cleaned.replace("--", "")
            for word in cleaned.split():
                if word and word not in seen:
                    seen.add(word)
                    self.add_word(word)

    @property
    def oov_rate(self) -> float:
        """Fraction of word lookups that fell through to grapheme fallback."""
        if self._total_lookups == 0:
            return 0.0
        return self._oov_lookups / self._total_lookups

    @property
    def oov_words(self) -> List[str]:
        """List of words that required grapheme fallback."""
        return sorted(self._oov_words)

    def oov_report(self) -> str:
        """Human-readable OOV report."""
        lines = [
            f"OOV Report:",
            f"  Total word lookups:  {self._total_lookups}",
            f"  OOV (grapheme only): {self._oov_lookups}",
            f"  OOV rate:            {self.oov_rate:.4f}",
        ]
        if self._oov_words:
            lines.append(f"  OOV words ({len(self._oov_words)}):")
            for w in sorted(self._oov_words)[:50]:
                lines.append(f"    - {w}")
            if len(self._oov_words) > 50:
                lines.append(f"    ... and {len(self._oov_words) - 50} more")
        return "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------
    # Phoneme → Word segmentation
    # ------------------------------------------------------------------

    def phonemes_to_words(
        self,
        phoneme_ids: List[int],
        beam_width: int = 10,
    ) -> List[Tuple[List[str], float]]:
        """
        Convert a phoneme ID sequence into word-level hypotheses.

        Primary method: split on SIL tokens (reliable in this dataset,
        since SIL is inserted between every word during data preparation).

        Fallback: if SIL-based splitting yields no chunks (e.g. no SIL
        tokens present), uses DP-based lexicon segmentation over the
        full phoneme string.

        Returns a list of (word_list, score) tuples sorted best-first.
        Lower score = fewer edits = better.
        """
        # Filter out blanks
        cleaned = [p for p in phoneme_ids if p != BLANK_IDX]
        if not cleaned:
            return [([], 0.0)]

        # Primary: split on SIL tokens to get word-level phoneme chunks
        chunks = self._split_on_sil(cleaned)

        if chunks:
            # SIL-based segmentation (primary path)
            return self._segment_from_chunks(chunks, beam_width)
        else:
            # Fallback: DP-based lexicon segmentation without SIL
            return self._dp_segment(cleaned, beam_width)

    def _segment_from_chunks(
        self,
        chunks: List[List[int]],
        beam_width: int,
    ) -> List[Tuple[List[str], float]]:
        """Segment pre-split phoneme chunks into words via beam search."""
        chunk_candidates: List[List[Tuple[str, float]]] = []
        for chunk in chunks:
            candidates = self._find_word_candidates(chunk)
            if not candidates:
                # Fall back to <unk> token with a penalty
                candidates = [("<unk>", float(len(chunk)))]
            chunk_candidates.append(candidates)

        # Beam search over word sequence combinations
        beams: List[Tuple[List[str], float]] = [([], 0.0)]

        for candidates in chunk_candidates:
            new_beams: List[Tuple[List[str], float]] = []
            for words, score in beams:
                for cand_word, cand_cost in candidates:
                    new_beams.append((words + [cand_word], score + cand_cost))
            # Prune
            new_beams.sort(key=lambda x: x[1])
            beams = new_beams[:beam_width]

        return beams

    def _dp_segment(
        self,
        phoneme_ids: List[int],
        beam_width: int = 10,
        max_word_phones: int = 20,
        insertion_cost: float = 0.5,
    ) -> List[Tuple[List[str], float]]:
        """
        DP-based lexicon segmentation without relying on SIL boundaries.

        Uses dynamic programming to find the best word-level segmentation
        of the phoneme string, allowing small edit-distance mismatches
        and optional phoneme insertions/deletions.

        This is a Viterbi-style algorithm over the phoneme sequence:
          dp[i] = best (word_list, cost) ending at phoneme position i

        Parameters
        ----------
        phoneme_ids : list of int
            Phoneme sequence (no blanks, no SIL).
        beam_width : int
            Number of segmentation hypotheses to keep.
        max_word_phones : int
            Maximum phoneme span to consider for a single word.
        insertion_cost : float
            Cost per unmatched phoneme (deletion/insertion).

        Returns
        -------
        list of (word_list, cost) sorted best-first.
        """
        N = len(phoneme_ids)
        if N == 0:
            return [([], 0.0)]

        # dp[i] = list of (word_list, total_cost) for best segmentations
        # ending at position i (exclusive)
        dp: List[List[Tuple[List[str], float]]] = [[] for _ in range(N + 1)]
        dp[0] = [([], 0.0)]

        for i in range(N):
            if not dp[i]:
                continue

            # Try all word spans starting at position i
            for j in range(i + 1, min(i + max_word_phones + 1, N + 1)):
                span = phoneme_ids[i:j]
                candidates = self._find_word_candidates(span)

                if candidates:
                    for word, edit_cost in candidates[:5]:  # top 5
                        for prev_words, prev_cost in dp[i][:beam_width]:
                            new_cost = prev_cost + edit_cost
                            dp[j].append((prev_words + [word], new_cost))
                            # Prune dp[j]
                            if len(dp[j]) > beam_width * 3:
                                dp[j].sort(key=lambda x: x[1])
                                dp[j] = dp[j][:beam_width]

            # Allow skipping a phoneme (insertion/deletion)
            if i + 1 <= N:
                for prev_words, prev_cost in dp[i][:beam_width]:
                    dp[i + 1].append((prev_words, prev_cost + insertion_cost))
                    if len(dp[i + 1]) > beam_width * 3:
                        dp[i + 1].sort(key=lambda x: x[1])
                        dp[i + 1] = dp[i + 1][:beam_width]

        # Collect results from dp[N]
        results = dp[N]
        if not results:
            # Fallback: if nothing reached the end, find best partial
            for i in range(N, -1, -1):
                if dp[i]:
                    results = dp[i]
                    break

        if not results:
            return [(["<unk>"], float(N))]

        results.sort(key=lambda x: x[1])
        return results[:beam_width]

    def _split_on_sil(self, phoneme_ids: List[int]) -> List[List[int]]:
        """Split a phoneme ID sequence on SIL tokens into word chunks."""
        chunks: List[List[int]] = []
        current: List[int] = []
        for pid in phoneme_ids:
            if pid == BLANK_IDX:
                continue
            if pid == self.sil_idx:
                if current:
                    chunks.append(current)
                    current = []
            else:
                current.append(pid)
        if current:
            chunks.append(current)
        return chunks

    def _find_word_candidates(
        self,
        chunk: List[int],
        max_candidates: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Find lexicon words whose pronunciation best matches the given
        phoneme chunk.  Returns (word, edit_distance) pairs.
        """
        candidates: List[Tuple[str, float]] = []

        # Try exact match first (fast path)
        chunk_key = tuple(chunk)
        if chunk_key in self._pron_to_words:
            for w in self._pron_to_words[chunk_key]:
                candidates.append((w, 0.0))

        if candidates:
            return candidates[:max_candidates]

        # Fuzzy match: use length-indexed lookup to avoid scanning
        # all entries.  Only check pronunciations within ±max_edit_dist
        # of the chunk length, reducing O(V) to O(V_similar_length).
        best: List[Tuple[str, float]] = []
        seen_words: set = set()
        chunk_len = len(chunk)
        for pron_len in range(
            max(1, chunk_len - self.max_edit_dist),
            chunk_len + self.max_edit_dist + 1,
        ):
            for word, pron in self._length_index.get(pron_len, []):
                if word in seen_words:
                    continue
                dist = _edit_distance(chunk, pron)
                if dist <= self.max_edit_dist:
                    best.append((word, float(dist)))
                    seen_words.add(word)

        best.sort(key=lambda x: x[1])
        return best[:max_candidates]

    def get_words(self) -> List[str]:
        """Return all words in the lexicon."""
        return list(self.entries.keys())
