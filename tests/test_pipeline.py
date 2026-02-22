"""
Tests for the decoding pipeline modules.

These tests verify the new modules:
  - ngram_decoder (wrapper around lm_decoder C++ module)
  - llm_rescorer (OPT-style rescoring)
  - nbest_augmentation (word swapping)
  - evaluation (WER computation)
  - decode_pipeline (full orchestration)

Note: Tests that require the lm_decoder C++ module are skipped if
the module is not installed.

Run with:
    uv run python -m pytest tests/test_pipeline.py -v
    or
    uv run python tests/test_pipeline.py
"""

import importlib
import unittest
import numpy as np
import sys
import os

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


class TestRearrangeLogits(unittest.TestCase):
    """Test logit rearrangement from model order to Kaldi order."""

    def test_rearrange_2d(self):
        from neural_decoder.ngram_decoder import rearrange_logits

        # Create fake logits: model order [BLANK(0), phonemes(1-39), SIL(40)]
        T = 10
        logits = np.zeros((T, 41), dtype=np.float32)
        logits[:, 0] = 1.0  # BLANK
        logits[:, 40] = 2.0  # SIL
        logits[:, 1] = 3.0  # first phoneme (AA)

        rearranged = rearrange_logits(logits)

        # After rearrange: [BLANK(0), SIL(1), phonemes(2-40)]
        self.assertEqual(rearranged.shape, (T, 41))
        np.testing.assert_array_equal(rearranged[:, 0], 1.0)  # BLANK still first
        np.testing.assert_array_equal(rearranged[:, 1], 2.0)  # SIL moved to index 1
        np.testing.assert_array_equal(rearranged[:, 2], 3.0)  # AA moved to index 2

    def test_rearrange_3d(self):
        from neural_decoder.ngram_decoder import rearrange_logits

        logits = np.random.randn(2, 10, 41).astype(np.float32)
        rearranged = rearrange_logits(logits)

        self.assertEqual(rearranged.shape, (2, 10, 41))
        # BLANK stays at 0
        np.testing.assert_array_equal(rearranged[..., 0], logits[..., 0])
        # SIL moved from 40 to 1
        np.testing.assert_array_equal(rearranged[..., 1], logits[..., 40])
        # Phonemes shifted from 1-39 to 2-40
        np.testing.assert_array_equal(rearranged[..., 2:], logits[..., 1:40])


class TestNbestAugmentation(unittest.TestCase):
    """Test n-best augmentation."""

    def test_basic_augmentation(self):
        from neural_decoder.nbest_augmentation import augment_nbest

        nbest = [
            ("the cat sat on the mat", -10.0, -5.0),
            ("the dog sat on the mat", -11.0, -4.5),
            ("a cat sat on the mat", -12.0, -4.0),
        ]

        augmented = augment_nbest(nbest, top_candidates_to_augment=3)

        # Should have at least as many as original
        self.assertGreaterEqual(len(augmented), len(nbest))

        # All entries should be (str, float, float) tuples
        for sent, ac, lm in augmented:
            self.assertIsInstance(sent, str)
            self.assertIsInstance(ac, (int, float, np.floating))
            self.assertIsInstance(lm, (int, float, np.floating))

    def test_no_augmentation_single(self):
        from neural_decoder.nbest_augmentation import augment_nbest

        nbest = [("hello world", -5.0, -2.0)]
        augmented = augment_nbest(nbest)
        self.assertEqual(len(augmented), 1)

    def test_sorted_by_score(self):
        from neural_decoder.nbest_augmentation import augment_nbest

        nbest = [
            ("a b c", -10.0, -5.0),
            ("d e f", -11.0, -4.5),
        ]
        augmented = augment_nbest(nbest, acoustic_scale=0.3)

        # Result should be sorted by total score (descending)
        for i in range(len(augmented) - 1):
            s1 = 0.3 * augmented[i][1] + augmented[i][2]
            s2 = 0.3 * augmented[i + 1][1] + augmented[i + 1][2]
            self.assertGreaterEqual(s1 + 1e-9, s2)


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


class TestNgramDecoder(unittest.TestCase):
    """Test NgramWFSTDecoder (requires lm_decoder C++ module)."""

    def test_import_check(self):
        """Test that the module can be imported (even if lm_decoder is missing)."""
        try:
            from neural_decoder.ngram_decoder import NgramWFSTDecoder
        except ImportError:
            self.skipTest("lm_decoder C++ module not installed")

    def test_missing_path(self):
        """Test that a ValueError is raised for a missing path."""
        from neural_decoder.ngram_decoder import NgramWFSTDecoder

        with self.assertRaises((ValueError, ImportError)):
            NgramWFSTDecoder(lm_path="/nonexistent/path")


@unittest.skipUnless(_has_torch, "torch package not installed")
class TestLLMRescorer(unittest.TestCase):
    """Test LLM rescorer (basic structure tests, no model loading)."""

    def test_import(self):
        from neural_decoder.llm_rescorer import LLMRescorer

    def test_empty_nbest(self):
        from neural_decoder.llm_rescorer import LLMRescorer

        rescorer = LLMRescorer.__new__(LLMRescorer)
        rescorer.model_name = "gpt2"
        rescorer._model = None
        rescorer._tokenizer = None
        rescorer.device = "cpu"
        rescorer.torch_dtype = None
        rescorer.cache_dir = None

        # Calling rescore with empty list should handle gracefully
        result = rescorer.rescore([], acoustic_scale=0.3, alpha=0.5)
        self.assertEqual(result[0], "")


class TestDecodePipeline(unittest.TestCase):
    """Test pipeline config and structure."""

    def test_config_defaults(self):
        from neural_decoder.decode_pipeline import PipelineConfig

        config = PipelineConfig()
        self.assertEqual(config.acoustic_scale, 0.3)
        self.assertEqual(config.blank_penalty, 9.0)
        self.assertEqual(config.nbest, 100)
        self.assertTrue(config.do_llm)
        self.assertTrue(config.augment_nbest)

    def test_pipeline_requires_setup(self):
        from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig

        config = PipelineConfig(lm_path="/nonexistent")
        pipeline = DecodePipeline(config)

        with self.assertRaises(RuntimeError):
            pipeline.decode(np.zeros((10, 41)))


if __name__ == "__main__":
    unittest.main()
