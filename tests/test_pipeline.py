"""
Comprehensive tests for the posterior-constrained decoding pipeline.

Tests each module (A1, A2, B1, B2, C1, C2, D, E) independently
and then the full pipeline end-to-end.

Run with:
    uv run python -m pytest tests/test_pipeline.py -v
    or
    uv run python tests/test_pipeline.py
"""

import math
import os
import sys
import unittest

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestPhonemeTable(unittest.TestCase):
    """Test phoneme_table.py constants and utilities."""

    def test_phone_def_length(self):
        from neural_decoder.phoneme_table import PHONE_DEF, PHONE_DEF_SIL

        self.assertEqual(len(PHONE_DEF), 39)
        self.assertEqual(len(PHONE_DEF_SIL), 40)

    def test_n_classes(self):
        from neural_decoder.phoneme_table import N_CLASSES_CTC, N_PHONEMES

        self.assertEqual(N_PHONEMES, 40)
        self.assertEqual(N_CLASSES_CTC, 41)

    def test_blank_idx(self):
        from neural_decoder.phoneme_table import BLANK_IDX

        self.assertEqual(BLANK_IDX, 0)

    def test_sil_idx(self):
        from neural_decoder.phoneme_table import SIL_IDX

        self.assertEqual(SIL_IDX, 40)

    def test_phone_to_id_mapping(self):
        from neural_decoder.phoneme_table import PHONE_TO_ID, ID_TO_PHONE

        # AA should be 1 (first phoneme, 1-indexed)
        self.assertEqual(PHONE_TO_ID["AA"], 1)
        self.assertEqual(PHONE_TO_ID["SIL"], 40)
        self.assertEqual(ID_TO_PHONE[1], "AA")
        self.assertEqual(ID_TO_PHONE[0], "<blank>")

    def test_phone_ids_to_str(self):
        from neural_decoder.phoneme_table import phone_ids_to_str

        result = phone_ids_to_str([1, 2, 3, 0, 40])
        self.assertIn("AA", result)
        self.assertNotIn("<blank>", result)

        result_with_blank = phone_ids_to_str([0, 1], skip_blank=False)
        self.assertIn("<blank>", result_with_blank)


class TestCTCBeamDecoder(unittest.TestCase):
    """Test ctc_beam_decoder.py – A1."""

    def _make_log_probs(self, T=20, C=41, seed=42):
        """Create synthetic log-softmax posteriors."""
        rng = np.random.RandomState(seed)
        logits = rng.randn(T, C).astype(np.float32)
        # Softmax
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        return np.log(probs + 1e-12)

    def test_decode_returns_list(self):
        from neural_decoder.ctc_beam_decoder import CTCBeamDecoder

        decoder = CTCBeamDecoder(beam_width=5, n_best=3)
        log_probs = self._make_log_probs()
        results = decoder.decode(log_probs)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

    def test_hypothesis_structure(self):
        from neural_decoder.ctc_beam_decoder import CTCBeamDecoder, CTCHypothesis

        decoder = CTCBeamDecoder(beam_width=5, n_best=3)
        log_probs = self._make_log_probs()
        results = decoder.decode(log_probs)
        for hyp in results:
            self.assertIsInstance(hyp, CTCHypothesis)
            self.assertIsInstance(hyp.phoneme_ids, list)
            self.assertTrue(isinstance(hyp.log_prob, (float, np.floating)))
            self.assertTrue(hyp.log_prob <= 0.0)
            self.assertIsInstance(hyp.frame_alignment, list)
            self.assertEqual(len(hyp.frame_alignment), 20)

    def test_sorted_by_score(self):
        from neural_decoder.ctc_beam_decoder import CTCBeamDecoder

        decoder = CTCBeamDecoder(beam_width=10, n_best=5)
        log_probs = self._make_log_probs(T=30)
        results = decoder.decode(log_probs)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].log_prob, results[i + 1].log_prob)

    def test_no_blank_in_output(self):
        from neural_decoder.ctc_beam_decoder import CTCBeamDecoder

        decoder = CTCBeamDecoder(beam_width=10, blank=0)
        log_probs = self._make_log_probs()
        results = decoder.decode(log_probs)
        for hyp in results:
            self.assertNotIn(0, hyp.phoneme_ids)

    def test_blank_penalty(self):
        from neural_decoder.ctc_beam_decoder import CTCBeamDecoder

        log_probs = self._make_log_probs(T=15, seed=123)
        # With and without blank penalty should give different results
        d1 = CTCBeamDecoder(beam_width=5, blank_penalty=0.0, n_best=5)
        d2 = CTCBeamDecoder(beam_width=5, blank_penalty=5.0, n_best=5)
        r1 = d1.decode(log_probs)
        r2 = d2.decode(log_probs)
        # The n-best sets should differ (penalty changes beam exploration)
        r1_seqs = [tuple(h.phoneme_ids) for h in r1]
        r2_seqs = [tuple(h.phoneme_ids) for h in r2]
        # At least check they both produce output; penalty should change
        # which hypotheses survive in the beam
        self.assertGreater(len(r1), 0)
        self.assertGreater(len(r2), 0)

    def test_with_length(self):
        from neural_decoder.ctc_beam_decoder import CTCBeamDecoder

        decoder = CTCBeamDecoder(beam_width=5)
        log_probs = self._make_log_probs(T=30)
        # Decode with shorter length
        r_full = decoder.decode(log_probs, length=30)
        r_short = decoder.decode(log_probs, length=10)
        # Short should have different alignment length
        self.assertEqual(len(r_short[0].frame_alignment), 10)
        self.assertEqual(len(r_full[0].frame_alignment), 30)

    def test_decode_batch(self):
        from neural_decoder.ctc_beam_decoder import CTCBeamDecoder

        decoder = CTCBeamDecoder(beam_width=5, n_best=2)
        batch = np.stack([self._make_log_probs(seed=i) for i in range(3)])
        lengths = np.array([20, 15, 18])
        results = decoder.decode_batch(batch, lengths)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertGreater(len(r), 0)

    def test_peaked_distribution(self):
        """When posteriors are peaked, greedy and beam should agree."""
        from neural_decoder.ctc_beam_decoder import CTCBeamDecoder

        T, C = 10, 41
        # Create a strongly peaked distribution
        log_probs = np.full((T, C), -10.0, dtype=np.float32)
        # Frame 0,1: blank; Frame 2,3: phoneme 5; Frame 4: blank;
        # Frame 5,6,7: phoneme 10; Frame 8,9: blank
        pattern = [0, 0, 5, 5, 0, 10, 10, 10, 0, 0]
        for t, c in enumerate(pattern):
            log_probs[t, c] = -0.01
        # Renormalise
        for t in range(T):
            lse = np.log(np.sum(np.exp(log_probs[t])))
            log_probs[t] -= lse

        decoder = CTCBeamDecoder(beam_width=10, n_best=3)
        results = decoder.decode(log_probs)
        # Best hypothesis should be [5, 10] (collapse repeats, remove blanks)
        self.assertEqual(results[0].phoneme_ids, [5, 10])


class TestLexicon(unittest.TestCase):
    """Test lexicon.py – A2."""

    def test_build_from_sentences(self):
        from neural_decoder.lexicon import PronunciationLexicon

        lex = PronunciationLexicon(max_edit_dist=1)
        lex.build_from_sentences(["hello world", "good morning", "hello there"])
        self.assertGreater(lex.size, 0)
        self.assertIn("hello", lex.entries)
        self.assertIn("world", lex.entries)

    def test_add_word(self):
        from neural_decoder.lexicon import PronunciationLexicon

        lex = PronunciationLexicon()
        lex.add_word("test", pronunciation=[30, 10, 28, 30])  # T EH S T
        self.assertIn("test", lex.entries)
        self.assertEqual(len(lex.entries["test"].pronunciations), 1)

    def test_phonemes_to_words_exact(self):
        from neural_decoder.lexicon import PronunciationLexicon
        from neural_decoder.phoneme_table import SIL_IDX

        lex = PronunciationLexicon(max_edit_dist=0)
        # Add words with known pronunciations
        lex.add_word("hi", pronunciation=[15, 5])  # HH AY
        lex.add_word("there", pronunciation=[9, 10, 27])  # DH EH R
        # Input: HH AY SIL DH EH R SIL
        phoneme_ids = [15, 5, SIL_IDX, 9, 10, 27, SIL_IDX]
        results = lex.phonemes_to_words(phoneme_ids)
        self.assertGreater(len(results), 0)
        best_words, best_score = results[0]
        self.assertEqual(best_words, ["hi", "there"])
        self.assertEqual(best_score, 0.0)

    def test_phonemes_to_words_fuzzy(self):
        from neural_decoder.lexicon import PronunciationLexicon
        from neural_decoder.phoneme_table import SIL_IDX

        lex = PronunciationLexicon(max_edit_dist=1)
        lex.add_word("hi", pronunciation=[15, 5])
        # Input has a slight error: [15, 6] instead of [15, 5]
        phoneme_ids = [15, 6, SIL_IDX]
        results = lex.phonemes_to_words(phoneme_ids)
        self.assertGreater(len(results), 0)
        # Should still find "hi" with edit dist 1
        found_hi = any(w[0] == ["hi"] for w in results)
        self.assertTrue(found_hi)

    def test_split_on_sil(self):
        from neural_decoder.lexicon import PronunciationLexicon
        from neural_decoder.phoneme_table import SIL_IDX, BLANK_IDX

        lex = PronunciationLexicon()
        chunks = lex._split_on_sil([1, 2, SIL_IDX, 3, 4, SIL_IDX])
        self.assertEqual(chunks, [[1, 2], [3, 4]])
        # Blanks should be ignored
        chunks2 = lex._split_on_sil([BLANK_IDX, 1, SIL_IDX, BLANK_IDX, 2])
        self.assertEqual(chunks2, [[1], [2]])

    def test_empty_input(self):
        from neural_decoder.lexicon import PronunciationLexicon

        lex = PronunciationLexicon()
        results = lex.phonemes_to_words([])
        self.assertEqual(results, [([], 0.0)])

    def test_get_words(self):
        from neural_decoder.lexicon import PronunciationLexicon

        lex = PronunciationLexicon()
        lex.add_word("hello", pronunciation=[15, 10, 20, 24])
        lex.add_word("world", pronunciation=[35, 11, 20, 8])
        words = lex.get_words()
        self.assertIn("hello", words)
        self.assertIn("world", words)


class TestUncertainty(unittest.TestCase):
    """Test uncertainty.py – B1."""

    def _make_log_probs(self, T=20, C=41, peaked=False):
        rng = np.random.RandomState(42)
        if peaked:
            log_probs = np.full((T, C), -10.0, dtype=np.float32)
            for t in range(T):
                log_probs[t, rng.randint(0, C)] = -0.01
        else:
            logits = rng.randn(T, C).astype(np.float32)
            logits -= logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
            log_probs = np.log(probs + 1e-12)
        return log_probs

    def test_frame_uncertainty_shape(self):
        from neural_decoder.uncertainty import UncertaintyEstimator

        est = UncertaintyEstimator()
        lp = self._make_log_probs()
        info = est.compute_frame_uncertainty(lp)
        self.assertEqual(info.entropy.shape, (20,))
        self.assertEqual(info.margin.shape, (20,))
        self.assertEqual(info.blank_prob.shape, (20,))
        self.assertEqual(info.argmax.shape, (20,))
        self.assertEqual(info.length, 20)

    def test_frame_uncertainty_with_length(self):
        from neural_decoder.uncertainty import UncertaintyEstimator

        est = UncertaintyEstimator()
        lp = self._make_log_probs(T=30)
        info = est.compute_frame_uncertainty(lp, length=15)
        self.assertEqual(info.length, 15)
        self.assertEqual(info.entropy.shape, (15,))

    def test_entropy_nonnegative(self):
        from neural_decoder.uncertainty import UncertaintyEstimator

        est = UncertaintyEstimator()
        lp = self._make_log_probs()
        info = est.compute_frame_uncertainty(lp)
        self.assertTrue(np.all(info.entropy >= 0))

    def test_peaked_low_entropy(self):
        from neural_decoder.uncertainty import UncertaintyEstimator

        est = UncertaintyEstimator()
        lp = self._make_log_probs(peaked=True)
        info = est.compute_frame_uncertainty(lp)
        # Peaked dist should have low entropy
        self.assertTrue(np.mean(info.entropy) < 1.0)

    def test_word_confidence(self):
        from neural_decoder.uncertainty import UncertaintyEstimator

        est = UncertaintyEstimator()
        lp = self._make_log_probs(T=30)
        info = est.compute_frame_uncertainty(lp)
        word_spans = [("hello", 0, 15), ("world", 15, 30)]
        confs = est.compute_word_confidence(info, word_spans)
        self.assertEqual(len(confs), 2)
        for wc in confs:
            self.assertGreaterEqual(wc.confidence, 0.0)
            self.assertLessEqual(wc.confidence, 1.0)

    def test_classify_word_confidence(self):
        from neural_decoder.uncertainty import UncertaintyEstimator, WordConfidence

        est = UncertaintyEstimator()
        wcs = [
            WordConfidence("hi", 0, 5, 0.5, 2.0, 0.1, 0.9),
            WordConfidence("um", 5, 10, 3.0, 0.1, 0.8, 0.2),
        ]
        classified = est.classify_word_confidence(wcs)
        self.assertEqual(classified[0][1], "high")
        self.assertEqual(classified[1][1], "low")


class TestConstrainedDecode(unittest.TestCase):
    """Test constrained_decode.py – B2 + C2."""

    def test_build_template(self):
        from neural_decoder.constrained_decode import ConstrainedHypothesisBuilder

        builder = ConstrainedHypothesisBuilder(high_confidence_threshold=0.7)

        word_hyps = [
            (["hello", "world"], -5.0),
            (["hello", "earth"], -6.0),
            (["hi", "world"], -7.0),
        ]
        word_confs = [("hello", 0.9), ("world", 0.3)]

        template = builder.build(word_hyps, word_confs)
        self.assertEqual(len(template.slots), 2)
        # "hello" should be locked (confidence 0.9 > 0.7)
        self.assertTrue(template.slots[0].is_locked)
        self.assertEqual(template.slots[0].locked_word, "hello")
        # "world" should be open (confidence 0.3 < 0.7)
        self.assertFalse(template.slots[1].is_locked)
        self.assertIn("world", template.slots[1].candidates)
        self.assertIn("earth", template.slots[1].candidates)

    def test_template_enumerate(self):
        from neural_decoder.constrained_decode import ConstrainedHypothesisBuilder

        builder = ConstrainedHypothesisBuilder(high_confidence_threshold=0.7)
        word_hyps = [
            (["hello", "world"], -5.0),
            (["hello", "earth"], -6.0),
        ]
        word_confs = [("hello", 0.9), ("world", 0.3)]
        template = builder.build(word_hyps, word_confs)
        combos = template.enumerate_candidates(max_combinations=100)
        # Should have "hello world" and "hello earth"
        self.assertTrue(any(c == ["hello", "world"] for c in combos))
        self.assertTrue(any(c == ["hello", "earth"] for c in combos))

    def test_get_best_candidate(self):
        from neural_decoder.constrained_decode import ConstrainedHypothesisBuilder

        builder = ConstrainedHypothesisBuilder()
        word_hyps = [(["a", "b", "c"], -2.0)]
        word_confs = [("a", 0.8), ("b", 0.9), ("c", 0.5)]
        template = builder.build(word_hyps, word_confs)
        best = template.get_best_candidate_sentence()
        self.assertIsInstance(best, str)
        self.assertIn("a", best)

    def test_slot_filling_decoder(self):
        from neural_decoder.constrained_decode import (
            ConstrainedHypothesisBuilder,
            SlotFillingDecoder,
        )

        # Use a dummy LLM scorer
        def dummy_scorer(text):
            # Prefer "hello world" over anything else
            if "hello world" in text:
                return -1.0
            return -5.0

        builder = ConstrainedHypothesisBuilder(high_confidence_threshold=0.7)
        word_hyps = [
            (["hello", "world"], -5.0),
            (["hello", "earth"], -6.0),
        ]
        word_confs = [("hello", 0.9), ("world", 0.3)]
        template = builder.build(word_hyps, word_confs)

        decoder = SlotFillingDecoder(
            llm_score_fn=dummy_scorer,
            lambda_neural=1.0,
            lambda_lm=1.0,
        )
        best_words, score = decoder.decode(template)
        self.assertEqual(best_words[0], "hello")  # locked
        self.assertIsInstance(score, float)

    def test_rescore_nbest(self):
        from neural_decoder.constrained_decode import (
            ConstrainedHypothesisBuilder,
            SlotFillingDecoder,
        )

        def dummy_scorer(text):
            if "hello world" in text:
                return -1.0
            return -5.0

        builder = ConstrainedHypothesisBuilder(high_confidence_threshold=0.7)
        word_hyps = [
            (["hello", "world"], -5.0),
            (["hello", "earth"], -6.0),
            (["goodbye", "world"], -7.0),
        ]
        word_confs = [("hello", 0.9), ("world", 0.3)]
        template = builder.build(word_hyps, word_confs)

        decoder = SlotFillingDecoder(
            llm_score_fn=dummy_scorer,
            lambda_neural=1.0,
            lambda_lm=1.0,
            gamma_constraint=10.0,
        )
        candidates = [wh[0] for wh in word_hyps]
        neural_scores = [wh[1] for wh in word_hyps]
        rescored = decoder.rescore_nbest(template, candidates, neural_scores)
        # "goodbye world" should be heavily penalised (edits locked slot)
        goodbye_idx = next(i for i, (w, _) in enumerate(rescored) if w[0] == "goodbye")
        hello_idx = next(i for i, (w, _) in enumerate(rescored) if "hello" in w)
        self.assertGreater(goodbye_idx, hello_idx)

    def test_empty_template(self):
        from neural_decoder.constrained_decode import (
            ConstrainedHypothesisBuilder,
            SlotFillingDecoder,
        )

        builder = ConstrainedHypothesisBuilder()
        template = builder.build([], [])
        self.assertEqual(len(template.slots), 0)

        decoder = SlotFillingDecoder(llm_score_fn=lambda x: 0.0)
        words, score = decoder.decode(template)
        self.assertEqual(words, [])


class TestLLMScorer(unittest.TestCase):
    """Test llm_scorer.py – C1."""

    def test_score_basic(self):
        from neural_decoder.llm_scorer import LLMScorer

        scorer = LLMScorer(model_name="gpt2", device="cpu")
        score = scorer.score("hello world")
        self.assertIsInstance(score, float)
        self.assertLess(score, 0.0)  # log-prob should be negative

    def test_score_empty(self):
        from neural_decoder.llm_scorer import LLMScorer

        scorer = LLMScorer(model_name="gpt2", device="cpu")
        score = scorer.score("")
        self.assertEqual(score, 0.0)

    def test_coherent_higher_than_gibberish(self):
        from neural_decoder.llm_scorer import LLMScorer

        scorer = LLMScorer(model_name="gpt2", device="cpu")
        s_coherent = scorer.score("the cat sat on the mat")
        s_gibberish = scorer.score("zxq plmb fnrk qwt yxz")
        self.assertGreater(s_coherent, s_gibberish)

    def test_score_batch(self):
        from neural_decoder.llm_scorer import LLMScorer

        scorer = LLMScorer(model_name="gpt2", device="cpu")
        scores = scorer.score_batch(["hello world", "goodbye world"])
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertIsInstance(s, float)

    def test_score_incremental(self):
        from neural_decoder.llm_scorer import LLMScorer

        scorer = LLMScorer(model_name="gpt2", device="cpu")
        scores = scorer.score_incremental(
            "the cat", ["sat on the mat", "flew to the moon"]
        )
        self.assertEqual(len(scores), 2)


class TestDistillation(unittest.TestCase):
    """Test distillation.py – D1, D2, D3."""

    def test_candidate_features(self):
        from neural_decoder.distillation import CandidateFeatures

        feat = CandidateFeatures(neural_score=-5.0, lm_score=-3.0, n_words=3)
        arr = feat.to_numpy()
        self.assertEqual(arr.shape, (9,))
        self.assertEqual(arr[0], -5.0)

    def test_distillation_data_collector(self):
        from neural_decoder.distillation import DistillationDataCollector

        collector = DistillationDataCollector(
            llm_score_fn=lambda x: -float(len(x)) / 10.0,
        )
        sample = collector.collect_sample(
            candidates=[["hello", "world"], ["hi", "world"]],
            neural_scores=[-5.0, -6.0],
        )
        self.assertEqual(len(sample.candidates), 2)
        self.assertEqual(len(sample.features), 2)
        self.assertEqual(len(sample.teacher_scores), 2)

    def test_pair_dataset(self):
        from neural_decoder.distillation import (
            DistillationDataCollector,
            DistillationDataset,
            DistillationSample,
        )

        collector = DistillationDataCollector(
            llm_score_fn=lambda x: -float(len(x)) / 10.0,
        )
        sample = collector.collect_sample(
            candidates=[["hello", "world"], ["hi", "world"], ["goodbye", "earth"]],
            neural_scores=[-5.0, -6.0, -7.0],
        )
        dataset = DistillationDataset(samples=[sample])
        pairs = dataset.get_pair_dataset()
        self.assertGreater(len(pairs), 0)
        for best_f, other_f, margin in pairs:
            self.assertEqual(best_f.shape, (9,))
            self.assertEqual(other_f.shape, (9,))

    def test_student_reranker(self):
        import torch
        from neural_decoder.distillation import StudentReranker, CandidateFeatures

        student = StudentReranker(feature_dim=9, hidden_dim=32)
        # Forward pass
        x = torch.randn(3, 9)
        out = student(x)
        self.assertEqual(out.shape, (3, 1))

    def test_train_student(self):
        from neural_decoder.distillation import (
            StudentReranker,
            DistillationDataset,
            DistillationSample,
            DistillationDataCollector,
            train_student,
        )

        collector = DistillationDataCollector(
            llm_score_fn=lambda x: -float(len(x)) / 10.0,
        )
        samples = []
        for i in range(5):
            sample = collector.collect_sample(
                candidates=[["word" + str(j)] for j in range(3)],
                neural_scores=[-5.0 - j for j in range(3)],
            )
            samples.append(sample)

        dataset = DistillationDataset(samples=samples)
        student = StudentReranker(feature_dim=9, hidden_dim=16)
        losses = train_student(student, dataset, n_epochs=20, verbose=False)
        self.assertGreater(len(losses), 0)

    def test_student_rerank(self):
        from neural_decoder.distillation import StudentReranker, CandidateFeatures

        student = StudentReranker(feature_dim=9, hidden_dim=16)
        candidates = [["hello", "world"], ["hi", "earth"]]
        feats = [
            CandidateFeatures(neural_score=-3.0),
            CandidateFeatures(neural_score=-5.0),
        ]
        best_words, score = student.rerank(candidates, feats)
        self.assertIsInstance(best_words, list)
        self.assertIsInstance(score, float)


class TestEvaluation(unittest.TestCase):
    """Test evaluation.py – E."""

    def test_wer(self):
        from neural_decoder.evaluation import evaluate_wer

        preds = [["hello", "world"], ["good"]]
        refs = [["hello", "world"], ["good", "morning"]]
        wer = evaluate_wer(preds, refs)
        # Second utterance has 1 deletion; total ref words = 2 + 2 = 4
        # WER = 1/4 = 0.25
        self.assertAlmostEqual(wer, 0.25, places=3)

    def test_cer(self):
        from neural_decoder.evaluation import evaluate_cer

        preds = ["hello", "gd"]
        refs = ["hello", "good"]
        cer = evaluate_cer(preds, refs)
        self.assertGreater(cer, 0.0)

    def test_per(self):
        from neural_decoder.evaluation import evaluate_per

        preds = [[1, 2, 3], [4, 5]]
        refs = [[1, 2, 3], [4, 6]]
        per = evaluate_per(preds, refs)
        # 1 substitution out of 5 total ref phones
        self.assertAlmostEqual(per, 1.0 / 5.0, places=3)

    def test_constraint_adherence_perfect(self):
        from neural_decoder.evaluation import constraint_adherence

        preds = [["hello", "world"]]
        cand_sets = [[["hello", "world"], ["hello", "earth"]]]
        ca = constraint_adherence(preds, cand_sets)
        self.assertEqual(ca, 1.0)

    def test_constraint_adherence_partial(self):
        from neural_decoder.evaluation import constraint_adherence

        preds = [["hello", "mars"]]
        cand_sets = [[["hello", "world"], ["hello", "earth"]]]
        ca = constraint_adherence(preds, cand_sets)
        # "hello" is in set, "mars" is not → 0.5
        self.assertAlmostEqual(ca, 0.5, places=3)

    def test_wer_by_confidence_bucket(self):
        from neural_decoder.evaluation import wer_by_confidence_bucket

        preds = ["hello", "world", "good", "bad"]
        refs = ["hello", "earth", "good", "good"]
        confs = [0.9, 0.2, 0.8, 0.1]
        buckets = wer_by_confidence_bucket(preds, refs, confs, n_buckets=2)
        self.assertEqual(len(buckets), 2)

    def test_eval_report_summary(self):
        from neural_decoder.evaluation import EvalReport

        report = EvalReport(name="test", wer=0.15, cer=0.10, per=0.12)
        summary = report.summary()
        self.assertIn("test", summary)
        self.assertIn("0.15", summary)

    def test_ablation_ladder(self):
        from neural_decoder.evaluation import EvalReport, run_ablation_ladder

        reports = [
            EvalReport(name="baseline", wer=0.30, cer=0.20, per=0.15),
            EvalReport(name="constrained", wer=0.20, cer=0.15, per=0.12),
        ]
        table = run_ablation_ladder(reports)
        self.assertIn("baseline", table)
        self.assertIn("constrained", table)


class TestDecodePipeline(unittest.TestCase):
    """Test decode_pipeline.py – full integration."""

    def _make_log_probs(self, T=50, C=41, seed=42):
        rng = np.random.RandomState(seed)
        logits = rng.randn(T, C).astype(np.float32)
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        return np.log(probs + 1e-12)

    def test_pipeline_neural_only(self):
        from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig

        config = PipelineConfig(
            beam_width=5,
            n_best=3,
            decode_mode="neural_only",
            llm_model_name="gpt2",
            llm_device="cpu",
        )
        pipe = DecodePipeline(config)
        pipe.setup(["hello world", "good morning", "how are you"])

        lp = self._make_log_probs()
        result = pipe.decode_utterance(lp, length=50)

        self.assertIsInstance(result.final_sentence, str)
        self.assertGreater(result.total_time, 0)
        self.assertGreater(len(result.phoneme_hypotheses), 0)

    def test_pipeline_unconstrained_rescore(self):
        from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig

        config = PipelineConfig(
            beam_width=5,
            n_best=3,
            decode_mode="unconstrained_rescore",
            llm_model_name="gpt2",
            llm_device="cpu",
        )
        pipe = DecodePipeline(config)
        pipe.setup(["hello world", "good morning"])

        lp = self._make_log_probs()
        result = pipe.decode_utterance(lp, length=50)

        self.assertIsInstance(result.final_sentence, str)
        self.assertGreater(result.llm_time, 0)

    def test_pipeline_constrained_rescore(self):
        from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig

        config = PipelineConfig(
            beam_width=5,
            n_best=3,
            decode_mode="constrained_rescore",
            llm_model_name="gpt2",
            llm_device="cpu",
        )
        pipe = DecodePipeline(config)
        pipe.setup(["hello world", "good morning"])

        lp = self._make_log_probs()
        result = pipe.decode_utterance(lp, length=50)

        self.assertIsInstance(result.final_sentence, str)
        self.assertIsNotNone(result.template)

    def test_pipeline_slot_filling(self):
        from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig

        config = PipelineConfig(
            beam_width=5,
            n_best=3,
            decode_mode="slot_filling",
            llm_model_name="gpt2",
            llm_device="cpu",
            slot_filling_beam=3,
        )
        pipe = DecodePipeline(config)
        pipe.setup(["hello world", "good morning"])

        lp = self._make_log_probs()
        result = pipe.decode_utterance(lp, length=50)

        self.assertIsInstance(result.final_sentence, str)

    def test_pipeline_timing(self):
        from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig

        config = PipelineConfig(
            beam_width=3,
            n_best=2,
            decode_mode="neural_only",
        )
        pipe = DecodePipeline(config)
        pipe.setup(["hello world"])

        lp = self._make_log_probs(T=20)
        result = pipe.decode_utterance(lp)

        self.assertGreater(result.ctc_decode_time, 0)
        self.assertGreater(result.lexicon_time, 0)
        self.assertGreater(result.uncertainty_time, 0)
        self.assertGreaterEqual(
            result.total_time, result.ctc_decode_time + result.lexicon_time
        )

    def test_pipeline_uncertainty_output(self):
        from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig

        config = PipelineConfig(beam_width=5, n_best=3, decode_mode="neural_only")
        pipe = DecodePipeline(config)
        pipe.setup(["the cat sat", "on the mat"])

        lp = self._make_log_probs()
        result = pipe.decode_utterance(lp)

        self.assertIsNotNone(result.frame_uncertainty)
        self.assertEqual(result.frame_uncertainty.length, 50)


class TestCTCForcedAlignment(unittest.TestCase):
    """Test forced alignment and forward-backward in ctc_beam_decoder.py."""

    def test_forced_align_peaked(self):
        """Forced alignment on a peaked distribution."""
        from neural_decoder.ctc_beam_decoder import ctc_forced_align

        T, C = 10, 41
        log_probs = np.full((T, C), -10.0, dtype=np.float32)
        # Pattern: blank blank 5 5 blank 10 10 10 blank blank
        pattern = [0, 0, 5, 5, 0, 10, 10, 10, 0, 0]
        for t, c in enumerate(pattern):
            log_probs[t, c] = -0.01
        for t in range(T):
            lse = np.log(np.sum(np.exp(log_probs[t])))
            log_probs[t] -= lse

        alignment, expanded = ctc_forced_align(log_probs, [5, 10], blank=0)
        self.assertEqual(len(alignment), T)
        # The alignment should have phoneme 5 and 10 present
        self.assertIn(5, alignment)
        self.assertIn(10, alignment)

    def test_forced_align_empty_seq(self):
        """Empty label sequence should give all-blank alignment."""
        from neural_decoder.ctc_beam_decoder import ctc_forced_align

        T, C = 5, 41
        log_probs = np.zeros((T, C), dtype=np.float32)
        alignment, expanded = ctc_forced_align(log_probs, [], blank=0)
        self.assertEqual(len(alignment), T)
        self.assertTrue(all(a == 0 for a in alignment))

    def test_forward_backward_sums_to_one(self):
        """Forward-backward posteriors should sum to ~1 across states per frame."""
        from neural_decoder.ctc_beam_decoder import ctc_forward_backward

        T, C = 15, 41
        rng = np.random.RandomState(42)
        logits = rng.randn(T, C).astype(np.float32)
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        log_probs = np.log(probs + 1e-12)

        gamma = ctc_forward_backward(log_probs, [5, 10, 15], blank=0)
        self.assertEqual(gamma.shape[0], T)
        # Each frame's posteriors should sum to approximately 1
        for t in range(T):
            row_sum = gamma[t].sum()
            self.assertAlmostEqual(row_sum, 1.0, places=2)

    def test_hypothesis_has_label_posteriors(self):
        """Beam decoder should produce label_posteriors for the 1-best hypothesis.

        Only the top-ranked hypothesis gets full forward-backward posteriors;
        lower-ranked hypotheses use cheap argmax alignment (label_posteriors=None).
        """
        from neural_decoder.ctc_beam_decoder import CTCBeamDecoder

        T, C = 10, 41
        log_probs = np.full((T, C), -10.0, dtype=np.float32)
        pattern = [0, 0, 5, 5, 0, 10, 10, 10, 0, 0]
        for t, c in enumerate(pattern):
            log_probs[t, c] = -0.01
        for t in range(T):
            lse = np.log(np.sum(np.exp(log_probs[t])))
            log_probs[t] -= lse

        decoder = CTCBeamDecoder(beam_width=5, n_best=3)
        results = decoder.decode(log_probs)
        # 1-best must have label_posteriors
        self.assertIsNotNone(results[0].label_posteriors)
        self.assertEqual(results[0].label_posteriors.shape[0], T)
        # Non-top hypotheses may have label_posteriors=None (optimization)
        for hyp in results[1:]:
            if hyp.label_posteriors is not None:
                self.assertEqual(hyp.label_posteriors.shape[0], T)


class TestDPSegmentation(unittest.TestCase):
    """Test DP-based lexicon segmentation (fallback when no SIL)."""

    def test_dp_segment_no_sil(self):
        """DP segmentation should work without SIL tokens."""
        from neural_decoder.lexicon import PronunciationLexicon

        lex = PronunciationLexicon(max_edit_dist=0, use_cmudict=False)
        lex.add_word("hi", pronunciation=[15, 5])
        lex.add_word("there", pronunciation=[9, 10, 27])

        # No SIL in the input — should use DP fallback
        phoneme_ids = [15, 5, 9, 10, 27]
        results = lex.phonemes_to_words(phoneme_ids)
        self.assertGreater(len(results), 0)
        # Should find "hi there" or at least the individual words
        best_words = results[0][0]
        self.assertGreater(len(best_words), 0)

    def test_sil_primary_still_works(self):
        """SIL-based splitting should still be the primary path."""
        from neural_decoder.lexicon import PronunciationLexicon
        from neural_decoder.phoneme_table import SIL_IDX

        lex = PronunciationLexicon(max_edit_dist=0, use_cmudict=False)
        lex.add_word("hi", pronunciation=[15, 5])
        lex.add_word("there", pronunciation=[9, 10, 27])

        phoneme_ids = [15, 5, SIL_IDX, 9, 10, 27]
        results = lex.phonemes_to_words(phoneme_ids)
        best_words, best_score = results[0]
        self.assertEqual(best_words, ["hi", "there"])


class TestCMUdictLexicon(unittest.TestCase):
    """Test CMUdict integration and OOV handling."""

    def test_oov_rate_tracking(self):
        from neural_decoder.lexicon import PronunciationLexicon

        lex = PronunciationLexicon(use_cmudict=False, use_grapheme_fallback=True)
        # Words that g2p should handle
        lex.add_word("hello")
        lex.add_word("world")
        self.assertGreaterEqual(lex._total_lookups, 0)
        self.assertIsInstance(lex.oov_rate, float)

    def test_oov_report(self):
        from neural_decoder.lexicon import PronunciationLexicon

        lex = PronunciationLexicon(use_cmudict=False, use_grapheme_fallback=True)
        lex.build_from_sentences(["hello world"])
        report = lex.oov_report()
        self.assertIn("OOV Report", report)
        self.assertIn("Total word lookups", report)

    def test_grapheme_fallback(self):
        from neural_decoder.lexicon import PronunciationLexicon

        lex = PronunciationLexicon(use_cmudict=False, use_grapheme_fallback=True)
        ids = lex._grapheme_spell_out("test")
        self.assertGreater(len(ids), 0)
        # Should have 4 phonemes for t-e-s-t
        self.assertEqual(len(ids), 4)

    def test_multiple_pronunciations(self):
        """Words should be able to have multiple pronunciations."""
        from neural_decoder.lexicon import PronunciationLexicon

        lex = PronunciationLexicon(use_cmudict=False)
        lex.add_word("test", pronunciation=[30, 10, 28, 30])
        lex.add_word("test", pronunciation=[30, 10, 28, 31])  # variant
        self.assertEqual(len(lex.entries["test"].pronunciations), 2)


class TestLearnedConfidence(unittest.TestCase):
    """Test learned logistic regression confidence model."""

    def test_handtuned_fallback(self):
        """When no model is fitted, should use hand-tuned weights."""
        from neural_decoder.uncertainty import UncertaintyEstimator

        est = UncertaintyEstimator()
        conf = est._compute_confidence(0.5, 2.0, 0.1)
        self.assertGreater(conf, 0.0)
        self.assertLess(conf, 1.0)

    def test_fit_confidence_model(self):
        """Should be able to fit a logistic regression on dev data."""
        from neural_decoder.uncertainty import UncertaintyEstimator

        est = UncertaintyEstimator()
        # Simulated dev data with clearer separation
        features = [
            (3.0, 0.2, 0.05, 10),  # high margin, low entropy → correct
            (2.5, 0.3, 0.1, 8),
            (2.8, 0.25, 0.08, 12),
            (0.05, 3.5, 0.9, 5),  # low margin, high entropy → incorrect
            (0.1, 3.0, 0.85, 6),
            (0.08, 3.2, 0.88, 4),
        ]
        correct = [True, True, True, False, False, False]

        result = est.fit_confidence_model(features, correct)
        self.assertGreater(result["accuracy"], 0.5)
        self.assertIsNotNone(result["coefficients"])
        self.assertEqual(result["n_samples"], 6)

        # Now the learned model should be used instead of hand-tuned
        self.assertIsNotNone(est._confidence_model)
        conf = est._compute_confidence(3.0, 0.2, 0.05)
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_posterior_word_confidence(self):
        """Test word confidence from forward-backward posteriors."""
        from neural_decoder.uncertainty import UncertaintyEstimator
        from neural_decoder.ctc_beam_decoder import ctc_forward_backward

        T, C = 15, 41
        rng = np.random.RandomState(42)
        logits = rng.randn(T, C).astype(np.float32)
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        log_probs = np.log(probs + 1e-12)

        est = UncertaintyEstimator()
        frame_info = est.compute_frame_uncertainty(log_probs)

        label_seq = [5, 10]
        gamma = ctc_forward_backward(log_probs, label_seq, blank=0)

        word_confs = est.compute_word_confidence_from_posteriors(
            frame_info,
            gamma,
            label_seq,
            ["word1", "word2"],
            phoneme_to_word_map=[0, 1],
        )
        self.assertEqual(len(word_confs), 2)
        for wc in word_confs:
            self.assertGreaterEqual(wc.confidence, 0.0)
            self.assertLessEqual(wc.confidence, 1.0)


class TestTranscriptAdherence(unittest.TestCase):
    """Test new evaluation metrics."""

    def test_transcript_adherence_perfect(self):
        from neural_decoder.evaluation import transcript_adherence

        preds = [["hello", "world"]]
        cands = [[["hello", "world"], ["hello", "earth"]]]
        ta = transcript_adherence(preds, cands)
        self.assertEqual(ta, 1.0)

    def test_transcript_adherence_none(self):
        from neural_decoder.evaluation import transcript_adherence

        preds = [["hello", "mars"]]
        cands = [[["hello", "world"], ["hello", "earth"]]]
        ta = transcript_adherence(preds, cands)
        self.assertEqual(ta, 0.0)

    def test_slot_adherence(self):
        from neural_decoder.evaluation import slot_adherence

        preds = [["hello", "world"]]
        # Per-slot candidates
        slot_cands = [[["hello", "hi"], ["world", "earth"]]]
        sa = slot_adherence(preds, slot_cands)
        self.assertEqual(sa, 1.0)

    def test_slot_adherence_partial(self):
        from neural_decoder.evaluation import slot_adherence

        preds = [["hello", "mars"]]
        slot_cands = [[["hello", "hi"], ["world", "earth"]]]
        sa = slot_adherence(preds, slot_cands)
        self.assertAlmostEqual(sa, 0.5, places=3)

    def test_eval_report_new_fields(self):
        from neural_decoder.evaluation import EvalReport

        report = EvalReport(
            name="test",
            wer=0.15,
            transcript_adherence=0.95,
            hallucination_rate_word=0.02,
            hallucination_rate_transcript=0.05,
        )
        summary = report.summary()
        self.assertIn("transcript", summary.lower())


class TestTransformerStudent(unittest.TestCase):
    """Test TransformerStudentReranker."""

    def test_forward_pass(self):
        import torch
        from neural_decoder.distillation import TransformerStudentReranker

        student = TransformerStudentReranker(
            vocab_size=100,
            d_model=32,
            n_heads=2,
            n_layers=2,
            feature_dim=9,
        )
        token_ids = torch.randint(0, 100, (3, 10))
        mask = torch.ones(3, 10, dtype=torch.bool)
        features = torch.randn(3, 9)
        scores = student(token_ids, mask, features)
        self.assertEqual(scores.shape, (3, 1))

    def test_forward_without_features(self):
        import torch
        from neural_decoder.distillation import TransformerStudentReranker

        student = TransformerStudentReranker(
            vocab_size=100,
            d_model=32,
            n_heads=2,
            n_layers=2,
            feature_dim=0,
        )
        # Override head to accept d_model only
        student.head = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        token_ids = torch.randint(0, 100, (2, 8))
        scores = student(token_ids)
        self.assertEqual(scores.shape, (2, 1))


class TestIncrementalScoring(unittest.TestCase):
    """Test true incremental LLM scoring with KV cache."""

    def test_incremental_scores_continuation_only(self):
        """Incremental scoring should score continuation tokens only."""
        from neural_decoder.llm_scorer import LLMScorer

        scorer = LLMScorer(model_name="gpt2", device="cpu")
        # Full score of "the cat sat on the mat"
        full_score = scorer.score("the cat sat on the mat")
        # Incremental: continuation-only score
        inc_scores = scorer.score_incremental("the cat", ["sat on the mat"])
        # The incremental score should be different from the full score
        # (it only scores continuation tokens, not the prefix)
        self.assertEqual(len(inc_scores), 1)
        self.assertIsInstance(inc_scores[0], float)
        # Full score averages over all tokens; incremental over continuation only
        self.assertNotEqual(inc_scores[0], full_score)

    def test_incremental_empty_prefix(self):
        from neural_decoder.llm_scorer import LLMScorer

        scorer = LLMScorer(model_name="gpt2", device="cpu")
        scores = scorer.score_incremental("", ["hello world"])
        self.assertEqual(len(scores), 1)
        self.assertIsInstance(scores[0], float)

    def test_incremental_multiple_continuations(self):
        from neural_decoder.llm_scorer import LLMScorer

        scorer = LLMScorer(model_name="gpt2", device="cpu")
        scores = scorer.score_incremental(
            "the cat",
            ["sat on the mat", "flew to the moon", ""],
        )
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores[2], 0.0)  # empty continuation


if __name__ == "__main__":
    unittest.main()
