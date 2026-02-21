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
import os
import pickle
import re
import sys
import time

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
        device=device,
    ).to(device)

    weights_path = os.path.join(model_dir, "modelWeights")
    state_dict = torch.load(weights_path, map_location=device)
    # Use strict=False for cross-version compatibility (v2 adds new params)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(
            f"  Note: {len(missing)} new params initialized from scratch (architecture upgrade)"
        )
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys in checkpoint")
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

    for day_idx in range(len(loaded_data[data_partition])):
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

    for utt in utterances:
        result = pipeline.decode_utterance(utt["log_probs"], utt["length"])

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
    parser.add_argument("--beam_width", type=int, default=25)
    parser.add_argument("--n_best", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        default="constrained_rescore",
        choices=[
            "neural_only",
            "unconstrained_rescore",
            "constrained_rescore",
            "slot_filling",
        ],
        help="Decoding mode",
    )
    parser.add_argument(
        "--ablation", action="store_true", help="Run full ablation ladder"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results"
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

    # Setup pipeline
    config = PipelineConfig(
        beam_width=input_args.beam_width,
        n_best=input_args.n_best,
        llm_model_name=input_args.llm_model,
        llm_device=input_args.llm_device,
        decode_mode=input_args.mode,
    )

    pipeline = DecodePipeline(config)
    pipeline.setup(training_sentences)
    print(f"  Lexicon size: {pipeline.lexicon.size} words")

    if input_args.ablation:
        # Run full ablation ladder
        print("\n--- Running ablation ladder ---")
        reports = []

        modes = [
            ("1. Neural-only (greedy CTC)", "neural_only"),
            ("2. + Unconstrained LLM rescore", "unconstrained_rescore"),
            ("3. + Posterior-constrained LLM", "constrained_rescore"),
            ("4. + Slot-filling LLM", "slot_filling"),
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
