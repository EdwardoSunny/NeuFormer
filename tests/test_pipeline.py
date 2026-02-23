"""
Tests for the neural decoder modules.

These tests verify:
  - phoneme_table (constants and mappings)
  - ngram_decoder (CTCBeamSearchDecoder, GreedyCTCDecoder, helpers)
  - evaluation (WER computation)

Run with:
    uv run python -m pytest tests/test_pipeline.py -v
    or
    uv run python tests/test_pipeline.py
"""

import importlib
import importlib.util
import os
import tempfile
import unittest

import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_has_editdistance = importlib.util.find_spec("editdistance") is not None
_has_torch = importlib.util.find_spec("torch") is not None


class TestPhonemeTable(unittest.TestCase):
    """Test phoneme_table.py constants."""

    def test_phoneme_count(self):
        from neural_decoder.phoneme_table import PHONE_DEF, PHONE_DEF_SIL, N_CLASSES_CTC

        self.assertEqual(len(PHONE_DEF), 39)
        self.assertEqual(len(PHONE_DEF_SIL), 40)
        self.assertEqual(N_CLASSES_CTC, 41)

    def test_blank_and_sil_indices(self):
        from neural_decoder.phoneme_table import BLANK_IDX, SIL_IDX

        self.assertEqual(BLANK_IDX, 0)
        self.assertEqual(SIL_IDX, 40)

    def test_phone_to_id_mapping(self):
        from neural_decoder.phoneme_table import PHONE_TO_ID, ID_TO_PHONE

        self.assertEqual(PHONE_TO_ID["AA"], 1)
        self.assertEqual(PHONE_TO_ID["SIL"], 40)
        self.assertEqual(ID_TO_PHONE[0], "<blank>")
        self.assertEqual(ID_TO_PHONE[1], "AA")
        self.assertEqual(ID_TO_PHONE[40], "SIL")


class TestNgramDecoderHelpers(unittest.TestCase):
    """Test helper functions in ngram_decoder.py."""

    def test_tokens_list(self):
        from neural_decoder.ngram_decoder import TOKENS, BLANK_TOKEN, SIL_TOKEN

        self.assertEqual(len(TOKENS), 41)
        self.assertEqual(TOKENS[0], BLANK_TOKEN)
        self.assertEqual(TOKENS[-1], SIL_TOKEN)
        self.assertEqual(TOKENS[1], "AA")
        self.assertEqual(TOKENS[39], "ZH")

    def test_write_tokens_file(self):
        from neural_decoder.ngram_decoder import _write_tokens_file, TOKENS

        with tempfile.TemporaryDirectory() as tmpdir:
            tok_path = os.path.join(tmpdir, "tokens.txt")
            _write_tokens_file(tok_path)

            with open(tok_path) as f:
                lines = [line.strip() for line in f.readlines()]

            self.assertEqual(len(lines), 41)
            for i, tok in enumerate(TOKENS):
                self.assertEqual(lines[i], tok)

    def test_convert_numeric_lexicon(self):
        from neural_decoder.ngram_decoder import (
            _convert_numeric_lexicon,
            SIL_TOKEN,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "lexicon_numbers.txt")
            dst = os.path.join(tmpdir, "lexicon.txt")

            with open(src, "w") as f:
                # word id1 id2 ...
                # 1=AA, 7=B, 16=HH, 12=ER
                f.write("TEST 1 7\n")  # AA B
                f.write("HI 16 1 40\n")  # HH AA SIL

            _convert_numeric_lexicon(src, dst)

            with open(dst) as f:
                lines = f.readlines()

            # TEST should be: TEST AA B |
            self.assertIn("TEST", lines[0])
            self.assertTrue(lines[0].strip().endswith(SIL_TOKEN))
            # HI should be: HI HH AA |  (SIL at end, 40 maps to |)
            self.assertIn("HI", lines[1])
            self.assertTrue(lines[1].strip().endswith(SIL_TOKEN))

    def test_find_lm_missing(self):
        from neural_decoder.ngram_decoder import _find_lm

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _find_lm(tmpdir)
            self.assertIsNone(result)

    def test_find_lm_present(self):
        from neural_decoder.ngram_decoder import _find_lm

        with tempfile.TemporaryDirectory() as tmpdir:
            arpa_path = os.path.join(tmpdir, "lm.arpa")
            with open(arpa_path, "w") as f:
                f.write("fake arpa\n")
            result = _find_lm(tmpdir)
            self.assertEqual(result, arpa_path)


@unittest.skipUnless(_has_torch, "torch package not installed")
class TestGreedyCTCDecoder(unittest.TestCase):
    """Test the GreedyCTCDecoder."""

    def test_greedy_decode_simple(self):
        import torch
        from neural_decoder.ngram_decoder import GreedyCTCDecoder, TOKENS

        decoder = GreedyCTCDecoder(tokens=TOKENS, blank=0)

        # Create a simple emission: 10 time steps, 41 classes
        # Make it predict blank mostly, with a few phoneme spikes
        emission = torch.full((10, 41), -10.0)
        emission[:, 0] = 0.0  # blank is dominant

        # Force a few phoneme predictions (non-blank)
        # Index 16 = HH, index 1 = AA (should produce "HH AA" -> no | so one "word")
        emission[2, 0] = -10.0
        emission[2, 16] = 0.0  # HH
        emission[5, 0] = -10.0
        emission[5, 1] = 0.0  # AA

        words = decoder(emission)
        # Should decode to phonemes joined without |, so one "word" of "HHAA"
        self.assertIsInstance(words, list)
        self.assertTrue(len(words) > 0)

    def test_greedy_decode_with_sil(self):
        import torch
        from neural_decoder.ngram_decoder import GreedyCTCDecoder, TOKENS

        decoder = GreedyCTCDecoder(tokens=TOKENS, blank=0)

        # Create emission with a word boundary (SIL = |, index 40)
        emission = torch.full((10, 41), -10.0)
        emission[:, 0] = 0.0  # blank dominant

        emission[1, 0] = -10.0
        emission[1, 16] = 0.0  # HH
        emission[3, 0] = -10.0
        emission[3, 40] = 0.0  # | (word boundary)
        emission[5, 0] = -10.0
        emission[5, 1] = 0.0  # AA

        words = decoder(emission)
        # Should produce two "words" separated by |
        self.assertIsInstance(words, list)
        self.assertEqual(len(words), 2)


@unittest.skipUnless(_has_editdistance, "editdistance package not installed")
class TestEvaluation(unittest.TestCase):
    """Test evaluation utilities."""

    def test_remove_punctuation(self):
        from neural_decoder.evaluation import remove_punctuation

        self.assertEqual(remove_punctuation("Hello, World!"), "hello world")
        self.assertEqual(remove_punctuation("it's a test."), "it's a test")
        self.assertEqual(remove_punctuation("foo--bar"), "foobar")

    def test_compute_wer_perfect(self):
        from neural_decoder.evaluation import compute_wer

        wer, ed, n_ref = compute_wer(
            ["the cat sat", "hello world"],
            ["the cat sat", "hello world"],
        )
        self.assertEqual(wer, 0.0)
        self.assertEqual(ed, 0)
        self.assertEqual(n_ref, 5)

    def test_compute_wer_all_wrong(self):
        from neural_decoder.evaluation import compute_wer

        wer, ed, n_ref = compute_wer(
            ["a b c"],
            ["d e f"],
        )
        self.assertEqual(wer, 1.0)
        self.assertEqual(ed, 3)

    def test_compute_wer_partial(self):
        from neural_decoder.evaluation import compute_wer

        wer, ed, n_ref = compute_wer(
            ["the dog sat"],
            ["the cat sat"],
        )
        self.assertAlmostEqual(wer, 1 / 3)
        self.assertEqual(ed, 1)

    def test_oracle_wer(self):
        from neural_decoder.evaluation import oracle_wer

        nbest_lists = [
            ["the cat sat", "the dog sat", "a cat sat"],
            ["hello world", "hello earth"],
        ]
        references = ["the cat sat", "hello world"]

        wer = oracle_wer(nbest_lists, references)
        self.assertEqual(wer, 0.0)

    def test_oracle_wer_imperfect(self):
        from neural_decoder.evaluation import oracle_wer

        nbest_lists = [
            ["the dog sat", "a cat ran"],  # best is "the dog sat" (1 error)
        ]
        references = ["the cat sat"]

        wer = oracle_wer(nbest_lists, references)
        self.assertAlmostEqual(wer, 1 / 3)


if __name__ == "__main__":
    unittest.main()
