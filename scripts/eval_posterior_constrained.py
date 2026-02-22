"""
Evaluation script using the posterior-constrained LLM decoding pipeline.

Replaces the old eval_competition.py (5-gram + OPT-6B) with the new
pipeline: CTC beam → lexicon → uncertainty → constrained LLM → student.

Usage
-----
    python scripts/eval_posterior_constrained.py \
        --model_path logs/speech_logs/conformer_improved \
        --dataset_path ptDecoder_ctc \
        --llm_model meta-llama/Meta-Llama-3-8B \
        --mode constrained_rescore
"""

import argparse
import math
import os
import pickle
import re
import sys
import time

from tqdm import tqdm

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_decoder.dataset import SpeechDataset
from neural_decoder.transformer_ctc import NeuralTransformerCTCModel
from neural_decoder.model import GRUDecoder
from neural_decoder.decode_pipeline import DecodePipeline, PipelineConfig, DecodeResult
from neural_decoder.evaluation import (
    EvalReport,
    evaluate_wer,
    evaluate_cer,
    evaluate_per,
    oracle_wer,
    constraint_adherence,
    transcript_adherence,
    wer_by_confidence_bucket,
    run_ablation_ladder,
)
from neural_decoder.phoneme_table import BLANK_IDX, SIL_IDX


def load_conformer_model(model_dir: str, n_days: int = 24, device: str = "cuda"):
    """Load a trained Conformer model."""
    with open(os.path.join(model_dir, "args"), "rb") as f:
        args = pickle.load(f)

    model = NeuralTransformerCTCModel(
        n_channels=args["nInputFeatures"],
        n_classes=args["nClasses"] + 1,
        n_days=n_days,
        frontend_dim=args.get("frontend_dim", 1024),
        latent_dim=args.get("latent_dim", 1024),
        autoencoder_hidden_dim=args.get("autoencoder_hidden_dim", 512),
        transformer_layers=args.get("transformer_num_layers", 8),
        transformer_heads=args.get("transformer_n_heads", 8),
        transformer_ff_dim=args.get("transformer_dim_ff", 2048),
        transformer_dropout=args.get("transformer_dropout", 0.3),
        temporal_kernel=args.get("temporal_kernel", 32),
        temporal_stride=args.get("temporal_stride", 4),
        gaussian_smooth_width=args.get("gaussian_smooth_width", 2.0),
        conformer_conv_kernel=args.get("conformer_conv_kernel", 31),
        use_spec_augment=False,  # Disable during eval
        drop_path_prob=0.0,  # Disable during eval
        autoencoder_residual=args.get("autoencoder_residual", False),
        use_rope=args.get("use_rope", False),
        device=device,
    ).to(device)

    weights_path = os.path.join(model_dir, "modelWeights")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, args


def load_gru_model(model_dir: str, n_days: int = 24, device: str = "cuda"):
    """Load a trained GRU model."""
    with open(os.path.join(model_dir, "args"), "rb") as f:
        args = pickle.load(f)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=n_days,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    weights_path = os.path.join(model_dir, "modelWeights")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, args


def extract_logits(model, args, data_partition, loaded_data, device="cuda"):
    """
    Run inference on all utterances and extract logits.

    Returns list of dicts with keys:
      'log_probs': np.ndarray [T, C]
      'length': int
      'true_phonemes': np.ndarray
      'transcription': str
    """
    results = []
    is_conformer = args.get("model_type", "gru_baseline") == "transformer_ctc"

    n_days = len(loaded_data[data_partition])
    for day_idx in tqdm(range(n_days), desc="Extracting logits", unit="day"):
        day_data = loaded_data[data_partition][day_idx]
        ds = SpeechDataset([day_data])
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=_padding
        )

        for j, (X, y, X_len, y_len, _) in enumerate(loader):
            X = X.to(device)
            X_len = X_len.to(device)
            day_t = torch.tensor([day_idx], dtype=torch.int64, device=device)

            with torch.no_grad():
                if is_conformer:
                    log_probs, out_lens, _ = model(X, day_t, X_len)
                    # log_probs: [T, B, C] → [T, C]
                    lp = log_probs[:, 0, :].cpu().numpy()
                    out_len = out_lens[0].cpu().item()
                else:
                    logits = model.forward(X, day_t)
                    out_lens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    lp = logits[0].log_softmax(-1).cpu().numpy()
                    out_len = out_lens[0].cpu().item()

            true_seq = y[0][: y_len[0]].numpy()

            # Get transcription
            transcript = ""
            if "transcriptions" in day_data and j < len(day_data["transcriptions"]):
                transcript = str(day_data["transcriptions"][j]).strip()
                transcript = re.sub(r"[^a-zA-Z\- ']", "", transcript)
                transcript = transcript.replace("--", "").lower()

            results.append(
                {
                    "log_probs": lp[:out_len],
                    "length": int(out_len),
                    "true_phonemes": true_seq,
                    "transcription": transcript,
                }
            )

    return results


def _padding(batch):
    X, y, X_lens, y_lens, days = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
    )


def get_training_sentences(loaded_data) -> list:
    """Extract all training transcriptions for lexicon building."""
    sentences = []
    for day_data in loaded_data["train"]:
        if "transcriptions" in day_data:
            for t in day_data["transcriptions"]:
                s = str(t).strip()
                s = re.sub(r"[^a-zA-Z\- ']", "", s).lower()
                s = s.replace("--", "")
                if s:
                    sentences.append(s)
    return sentences


def run_decode_mode(
    utterances: list,
    pipeline: DecodePipeline,
    mode_name: str,
    config: PipelineConfig,
) -> EvalReport:
    """Run a specific decode mode and compute metrics."""
    config_copy = PipelineConfig(**config.__dict__)
    pipeline.config = config_copy

    all_pred_words = []
    all_ref_words = []
    all_pred_phonemes = []
    all_ref_phonemes = []
    all_pred_sentences = []
    all_ref_sentences = []
    all_word_confs = []
    all_candidate_sets = []

    start_time = time.time()
    n_diag = 5  # Print diagnostics for first N utterances

    for utt_idx, utt in enumerate(tqdm(utterances, desc=mode_name, unit="utt")):
        result = pipeline.decode_utterance(utt["log_probs"], utt["length"])

        # Diagnostics: print score breakdown for first few utterances
        if utt_idx < n_diag and result.word_beam_hypotheses:
            ref_sentence = utt["transcription"]
            tqdm.write(f"\n  [diag {utt_idx}] ref: {ref_sentence}")
            for rank, wh in enumerate(result.word_beam_hypotheses[:3]):
                tqdm.write(
                    f'    #{rank}: "{" ".join(wh.words)}" '
                    f"ctc={wh.ctc_log_prob:.1f} "
                    f"ng={wh.ngram_log_prob:.2f} "
                    f"combined={wh.combined_score:.1f}"
                )

        # Predicted words
        pred_words = result.final_words
        ref_sentence = utt["transcription"]
        ref_words = ref_sentence.split() if ref_sentence else []

        all_pred_words.append(pred_words)
        all_ref_words.append(ref_words)

        # Predicted phonemes
        all_pred_phonemes.append(result.best_phoneme_ids)
        all_ref_phonemes.append(utt["true_phonemes"].tolist())

        # Sentences for CER
        all_pred_sentences.append(" ".join(pred_words))
        all_ref_sentences.append(ref_sentence)

        # Confidence per word (flatten)
        for wc in result.word_confidences:
            all_word_confs.append(wc.confidence)

        # Candidate set for adherence
        if result.word_hypotheses:
            all_candidate_sets.append([wh[0] for wh in result.word_hypotheses])
        else:
            all_candidate_sets.append([pred_words])

    elapsed = time.time() - start_time

    # Compute metrics
    wer = evaluate_wer(all_pred_words, all_ref_words)
    cer = evaluate_cer(all_pred_sentences, all_ref_sentences)
    per = evaluate_per(all_pred_phonemes, all_ref_phonemes)
    ower = oracle_wer(all_candidate_sets, all_ref_words)
    ca = constraint_adherence(all_pred_words, all_candidate_sets)
    ta = transcript_adherence(all_pred_words, all_candidate_sets)

    # Confidence breakdown
    flat_pred = []
    flat_ref = []
    for pw, rw in zip(all_pred_words, all_ref_words):
        for i, w in enumerate(pw):
            flat_pred.append(w)
            flat_ref.append(rw[i] if i < len(rw) else "")

    conf_breakdown = (
        wer_by_confidence_bucket(
            flat_pred,
            flat_ref,
            all_word_confs[: len(flat_pred)],
        )
        if all_word_confs
        else []
    )

    return EvalReport(
        name=mode_name,
        wer=wer,
        cer=cer,
        per=per,
        constraint_adherence=ca,
        transcript_adherence=ta,
        hallucination_rate_word=1.0 - ca,
        hallucination_rate_transcript=1.0 - ta,
        runtime_seconds=elapsed,
        confidence_breakdown=conf_breakdown,
        extra={"oracle_wer": ower},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Posterior-constrained LLM decoding evaluation"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to ptDecoder_ctc pickle"
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="test",
        choices=["test", "competition"],
        help="Data partition to evaluate on",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace LLM model name",
    )
    parser.add_argument("--llm_device", type=str, default="cpu", help="Device for LLM")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for neural model"
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=25,
        help="Beam width (paper: 18, default: 25 for more diversity)",
    )
    parser.add_argument(
        "--n_best",
        type=int,
        default=100,
        help="N-best list size (paper: 100 for offline)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.8, help="CTC weight (paper: 0.8)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="N-gram vs LLM mixture (paper: 0.5)"
    )
    parser.add_argument(
        "--blank_penalty",
        type=float,
        default=None,
        help="Blank penalty (default: log(2) for online, log(7) for offline)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=[
            "online",
            "offline",
            "neural_only",
            "unconstrained_rescore",
            "constrained_rescore",
            "slot_filling",
        ],
        help="Decoding mode (online=paper default, offline=paper offline+LLM)",
    )
    parser.add_argument(
        "--ablation", action="store_true", help="Run full ablation ladder"
    )
    parser.add_argument(
        "--normalize_scores",
        action="store_true",
        default=True,
        help="Z-score normalize neural/LM scores before combining",
    )
    parser.add_argument(
        "--no_normalize_scores",
        action="store_false",
        dest="normalize_scores",
        help="Disable z-score normalization",
    )
    parser.add_argument(
        "--two_pass_top_k",
        type=int,
        default=10,
        help="Top-K for second-pass incremental rescoring (0 to disable)",
    )
    parser.add_argument(
        "--lambda_neural", type=float, default=0.5, help="Neural score weight"
    )
    parser.add_argument("--lambda_lm", type=float, default=1.0, help="LM score weight")
    parser.add_argument(
        "--gamma_constraint", type=float, default=1.0, help="Constraint penalty"
    )
    parser.add_argument(
        "--length_penalty_beta", type=float, default=0.1, help="Length bonus"
    )
    parser.add_argument(
        "--high_confidence_threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for locking slots",
    )
    parser.add_argument(
        "--lambda_ngram", type=float, default=1.0, help="N-gram score weight"
    )
    parser.add_argument(
        "--ngram_weight",
        type=float,
        default=0.5,
        help="N-gram weight during lexicon beam search (shallow fusion)",
    )
    parser.add_argument(
        "--ngram_order", type=int, default=5, help="N-gram order (3 or 5)"
    )
    parser.add_argument(
        "--no_ngram",
        action="store_true",
        help="Disable N-gram LM entirely",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results"
    )
    parser.add_argument(
        "--fit_confidence",
        action="store_true",
        help="Fit learned confidence model on first 20%% of data before eval",
    )

    input_args = parser.parse_args()

    # Load data
    print("Loading dataset...")
    with open(input_args.dataset_path, "rb") as f:
        loaded_data = pickle.load(f)

    # Load model
    print("Loading model...")
    with open(os.path.join(input_args.model_path, "args"), "rb") as f:
        model_args = pickle.load(f)

    is_conformer = model_args.get("model_type", "gru_baseline") == "transformer_ctc"
    n_days = len(loaded_data["train"])

    if is_conformer:
        model, args = load_conformer_model(
            input_args.model_path, n_days, input_args.device
        )
    else:
        model, args = load_gru_model(input_args.model_path, n_days, input_args.device)

    # Extract logits
    print(f"Extracting logits from {input_args.partition} partition...")
    utterances = extract_logits(
        model, args, input_args.partition, loaded_data, input_args.device
    )
    print(f"  Got {len(utterances)} utterances")

    # Get training sentences for lexicon
    training_sentences = get_training_sentences(loaded_data)
    print(f"  Lexicon will be built from {len(training_sentences)} training sentences")

    # Determine blank penalty: mode-dependent default
    if input_args.blank_penalty is not None:
        blank_penalty = input_args.blank_penalty
    elif input_args.mode == "offline":
        blank_penalty = math.log(7)  # ≈ 1.9459
    else:
        blank_penalty = math.log(2)  # ≈ 0.693

    # Determine N-gram order: mode-dependent default
    ngram_order = input_args.ngram_order
    if input_args.mode == "online" and ngram_order == 5:
        ngram_order = 3  # online default is 3-gram

    # Setup pipeline
    config = PipelineConfig(
        decode_mode=input_args.mode,
        beam_width=input_args.beam_width,
        n_best=input_args.n_best,
        blank_penalty=blank_penalty,
        alpha=input_args.alpha,
        beta=input_args.beta,
        word_beam_width=input_args.beam_width,
        ngram_order=ngram_order,
        use_ngram=not input_args.no_ngram,
        ngram_weight=input_args.ngram_weight,
        # Offline-specific
        offline_ngram_order=5,
        offline_blank_penalty=math.log(7),
        offline_top_k=100,
        # LLM (only loaded for offline mode)
        llm_model_name=input_args.llm_model,
        llm_device=input_args.llm_device,
        llm_load_in_8bit=True,
        # Legacy params
        lambda_neural=input_args.lambda_neural,
        lambda_lm=input_args.lambda_lm,
        lambda_ngram=input_args.lambda_ngram,
        gamma_constraint=input_args.gamma_constraint,
        length_penalty_beta=input_args.length_penalty_beta,
        high_confidence_threshold=input_args.high_confidence_threshold,
        normalize_scores=input_args.normalize_scores,
        two_pass_top_k=input_args.two_pass_top_k,
    )

    pipeline = DecodePipeline(config)
    pipeline.setup(training_sentences)
    print(f"  Lexicon size: {pipeline.lexicon.size} words")

    # Optionally fit learned confidence model on a dev split
    if input_args.fit_confidence:
        print("\nFitting learned confidence model on first 20% of data...")
        n_dev = max(1, len(utterances) // 5)
        dev_utts = utterances[:n_dev]
        eval_utts = utterances[n_dev:]

        # Run neural-only decoding on dev split to get word confidences
        dev_features = []  # (min_margin, mean_entropy, blank_ratio, duration)
        dev_correct = []  # bool

        config_dev = PipelineConfig(**config.__dict__)
        config_dev.decode_mode = "neural_only"
        pipeline.config = config_dev

        for utt in dev_utts:
            result = pipeline.decode_utterance(utt["log_probs"], utt["length"])
            ref_words = utt["transcription"].split() if utt["transcription"] else []
            pred_words = result.final_words

            for i, wc in enumerate(result.word_confidences):
                duration = max(wc.end_frame - wc.start_frame, 1)
                dev_features.append(
                    (wc.min_margin, wc.mean_entropy, wc.blank_ratio, duration)
                )
                is_correct = i < len(ref_words) and wc.word == ref_words[i]
                dev_correct.append(is_correct)

        if dev_features:
            fit_result = pipeline.uncertainty.fit_confidence_model(
                dev_features, dev_correct
            )
            print(
                f"  Fitted on {fit_result['n_samples']} words, "
                f"accuracy={fit_result['accuracy']:.4f}"
            )
            if fit_result["coefficients"]:
                print(f"  Coefficients: {fit_result['coefficients']}")
                print(f"  Intercept: {fit_result['intercept']:.4f}")
        else:
            print("  No dev data available for fitting")
            eval_utts = utterances  # Use all data

        # Use remaining data for evaluation
        utterances = eval_utts
        print(f"  Evaluating on remaining {len(utterances)} utterances")

        # Restore config
        pipeline.config = config

    if input_args.ablation:
        # Run full ablation ladder
        print("\n--- Running ablation ladder ---")
        reports = []

        modes = [
            ("1. Neural-only (greedy CTC)", "neural_only"),
            ("2. Online (word-beam + 3-gram)", "online"),
            ("3. Offline (5-gram + LLM rescore)", "offline"),
        ]

        for name, mode in modes:
            print(f"\n  Running: {name}")
            config.decode_mode = mode
            report = run_decode_mode(utterances, pipeline, name, config)
            reports.append(report)
            ow = report.extra.get("oracle_wer", float("nan"))
            print(
                f"    WER={report.wer:.4f}  Oracle={ow:.4f}  CER={report.cer:.4f}  "
                f"PER={report.per:.4f}  Time={report.runtime_seconds:.1f}s"
            )

        print("\n" + run_ablation_ladder(reports))

        # Save reports
        if input_args.output:
            with open(input_args.output, "wb") as f:
                pickle.dump(reports, f)
            print(f"\nReports saved to {input_args.output}")

    else:
        # Single mode
        print(f"\n--- Running: {input_args.mode} ---")
        report = run_decode_mode(utterances, pipeline, input_args.mode, config)
        print(report.summary())

        if input_args.output:
            with open(input_args.output, "wb") as f:
                pickle.dump(report, f)


if __name__ == "__main__":
    main()
