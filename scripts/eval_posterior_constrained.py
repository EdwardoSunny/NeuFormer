"""
Evaluation script for the Conformer + WFST n-gram + LLM pipeline.

Takes a trained Conformer model, runs inference to get logits, then
decodes through the NEJM-style pipeline:
  Conformer → logits → rearrange → WFST n-gram → augment → OPT rescore

Usage
-----
    python scripts/eval_posterior_constrained.py \
        --model_path logs/speech_logs/conformer_improved \
        --dataset_path ptDecoder_ctc \
        --lm_path /path/to/lm_dir \
        --do_llm \
        --llm_model facebook/opt-6.7b
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
from neural_decoder.evaluation import compute_wer, remove_punctuation
from neural_decoder.ngram_decoder import rearrange_logits


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
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
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
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model, args


def extract_logits(model, args, data_partition, loaded_data, device="cuda"):
    """
    Run inference on all utterances and extract logits.

    Returns list of dicts with keys:
      'logits': np.ndarray [T, C] (raw logits, NOT log-softmax)
      'length': int
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
                    # Note: the conformer outputs log_softmax, but the
                    # WFST decoder (lm_decoder.DecodeNumpy) expects raw
                    # logits and applies log_softmax internally.
                    # So we pass the log_probs as-is; DecodeNumpy will
                    # apply log_softmax again, but since log_softmax of
                    # log_softmax is close to log_softmax for peaked
                    # distributions, this works in practice.
                    # For best results, modify the conformer to output
                    # raw logits, or use DecodeNumpyLogProbs.
                    lp = log_probs[:, 0, :].cpu().numpy()
                    out_len = out_lens[0].cpu().item()
                else:
                    logits = model.forward(X, day_t)
                    out_lens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    lp = logits[0].cpu().numpy()  # raw logits
                    out_len = out_lens[0].cpu().item()

            # Get transcription
            transcript = ""
            if "transcriptions" in day_data and j < len(day_data["transcriptions"]):
                transcript = str(day_data["transcriptions"][j]).strip()
                transcript = re.sub(r"[^a-zA-Z\- ']", "", transcript)
                transcript = transcript.replace("--", "").lower()

            results.append(
                {
                    "logits": lp[:out_len],
                    "length": int(out_len),
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Conformer + WFST n-gram + LLM pipeline"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to ptDecoder_ctc pickle",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="test",
        choices=["test", "competition"],
        help="Data partition to evaluate on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Conformer inference",
    )

    # ---- WFST N-gram decoder ----
    parser.add_argument(
        "--lm_path",
        type=str,
        required=True,
        help="Path to LM directory containing TLG.fst and words.txt",
    )
    parser.add_argument("--acoustic_scale", type=float, default=0.3)
    parser.add_argument("--blank_penalty", type=float, default=9.0)
    parser.add_argument("--beam", type=float, default=17.0)
    parser.add_argument("--lattice_beam", type=float, default=8.0)
    parser.add_argument("--nbest", type=int, default=100)
    parser.add_argument(
        "--rescore",
        action="store_true",
        default=True,
        help="Rescore with unpruned G.fst",
    )
    parser.add_argument("--no_rescore", action="store_false", dest="rescore")

    # ---- N-best augmentation ----
    parser.add_argument(
        "--no_augment", action="store_true", help="Disable n-best augmentation"
    )
    parser.add_argument("--top_candidates_to_augment", type=int, default=20)
    parser.add_argument("--score_penalty_percent", type=float, default=0.01)

    # ---- LLM rescoring ----
    parser.add_argument(
        "--do_llm", action="store_true", help="Enable LLM rescoring (OPT)"
    )
    parser.add_argument("--llm_model", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--llm_cache_dir", type=str, default=None)
    parser.add_argument("--llm_device", type=str, default="cuda")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="LLM weight [0-1]: higher = more LLM, lower = more n-gram",
    )

    # ---- Output ----
    parser.add_argument(
        "--output", type=str, default=None, help="Output CSV file for predictions"
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

    # Setup decode pipeline
    config = PipelineConfig(
        lm_path=input_args.lm_path,
        acoustic_scale=input_args.acoustic_scale,
        blank_penalty=input_args.blank_penalty,
        beam=input_args.beam,
        lattice_beam=input_args.lattice_beam,
        nbest=input_args.nbest,
        rescore=input_args.rescore,
        augment_nbest=not input_args.no_augment,
        top_candidates_to_augment=input_args.top_candidates_to_augment,
        score_penalty_percent=input_args.score_penalty_percent,
        do_llm=input_args.do_llm,
        llm_model_name=input_args.llm_model,
        llm_cache_dir=input_args.llm_cache_dir,
        llm_device=input_args.llm_device,
        llm_alpha=input_args.alpha,
    )

    pipeline = DecodePipeline(config)
    pipeline.setup()

    # Decode all utterances
    predictions = []
    references = []

    print("\nDecoding...")
    for utt in tqdm(utterances, desc="Decoding", unit="utt"):
        result = pipeline.decode(utt["logits"])

        predictions.append(result.sentence)
        references.append(utt["transcription"])

    # Compute WER
    has_refs = any(r.strip() for r in references)
    if has_refs:
        wer, total_ed, total_ref = compute_wer(predictions, references)
        print(f"\n{'=' * 50}")
        print(f"Results ({input_args.partition})")
        print(f"{'=' * 50}")
        print(f"Total utterances:     {len(utterances)}")
        print(f"Total reference words: {total_ref}")
        print(f"Total edit distance:   {total_ed}")
        print(f"Aggregate WER:         {100 * wer:.2f}%")
        print(f"{'=' * 50}")

        # Print per-sentence results for first 20
        print("\nSample results:")
        for i in range(min(20, len(predictions))):
            ref = remove_punctuation(references[i])
            pred = remove_punctuation(predictions[i])
            print(f"\n  [{i}] Ref:  {ref}")
            print(f"       Pred: {pred}")
    else:
        print(f"\nNo reference transcriptions available for {input_args.partition}.")
        print("Predictions:")
        for i in range(min(20, len(predictions))):
            print(f"  [{i}] {predictions[i]}")

    # Save predictions
    if input_args.output:
        import csv

        os.makedirs(os.path.dirname(input_args.output) or ".", exist_ok=True)
        with open(input_args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text"])
            for i, pred in enumerate(predictions):
                writer.writerow([i, pred])
        print(f"\nPredictions saved to {input_args.output}")


if __name__ == "__main__":
    main()
