"""
Sweep LM decoder parameters to find optimal WER.

Extracts logits once, then decodes with many parameter combinations.
Much faster than re-running the full eval script each time.

Usage
-----
    uv run python scripts/sweep_lm_params.py \
        --model_path logs/speech_logs/conformer_v2 \
        --dataset_path ptDecoder_ctc \
        --partition test \
        --lm_dir lm_dir
"""

import argparse
import itertools
import os
import pickle
import re
import sys
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_decoder.dataset import SpeechDataset
from neural_decoder.evaluation import compute_wer, remove_punctuation
from neural_decoder.model import GRUDecoder
from neural_decoder.ngram_decoder import CTCBeamSearchDecoder, TOKENS
from neural_decoder.transformer_ctc import NeuralTransformerCTCModel


# ------------------------------------------------------------------
# Reuse model loading and logit extraction from eval script
# ------------------------------------------------------------------
def load_conformer(model_dir, n_days=24, device="cuda"):
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
        use_spec_augment=False,
        drop_path_prob=0.0,
        autoencoder_residual=args.get("autoencoder_residual", False),
        use_rope=args.get("use_rope", False),
        device=device,
    ).to(device)
    state = torch.load(
        os.path.join(model_dir, "modelWeights"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model, args


def _collate(batch):
    X, y, xl, yl, d = zip(*batch)
    return (
        pad_sequence(list(X), batch_first=True),
        pad_sequence(list(y), batch_first=True),
        torch.stack(list(xl)),
        torch.stack(list(yl)),
        torch.stack(list(d)),
    )


def extract_logits(model, args, partition, data, device="cuda"):
    is_conf = args.get("model_type", "gru_baseline") == "transformer_ctc"
    results = []
    for day_idx in tqdm(range(len(data[partition])), desc="Logits", unit="day"):
        day = data[partition][day_idx]
        loader = DataLoader(
            SpeechDataset([day]),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        )
        for j, (X, y, X_len, y_len, _) in enumerate(loader):
            X, X_len = X.to(device), X_len.to(device)
            day_t = torch.tensor([day_idx], dtype=torch.int64, device=device)
            with torch.no_grad():
                if is_conf:
                    lp, ol, _ = model(X, day_t, X_len)
                    lp = lp[:, 0, :].cpu().numpy()
                    ol = ol[0].cpu().item()
                else:
                    logits = model.forward(X, day_t)
                    ol = int(((X_len - model.kernelLen) / model.strideLen)[0].item())
                    lp = logits[0].cpu().numpy()
            tx = ""
            if "transcriptions" in day and j < len(day["transcriptions"]):
                tx = re.sub(
                    r"[^a-zA-Z\- ']",
                    "",
                    str(day["transcriptions"][j]).strip(),
                )
                tx = tx.replace("--", "").lower()
            results.append({"logits": lp[:ol], "transcription": tx})
    return results, is_conf


def decode_and_score(utterances, is_conf, decoder):
    """Decode all utterances and return WER."""
    preds, refs = [], []
    for utt in utterances:
        text = decoder.decode_text(utt["logits"], is_log_probs=is_conf)
        preds.append(text)
        refs.append(utt["transcription"])
    wer, ed, nref = compute_wer(preds, refs)
    return wer, ed, nref, preds


def main():
    ap = argparse.ArgumentParser(description="Sweep LM decoder parameters")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--partition", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lm_dir", required=True)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer parameter combinations for a quick test",
    )
    args = ap.parse_args()

    # ---- load data + model ----
    print("Loading dataset ...")
    with open(args.dataset_path, "rb") as f:
        data = pickle.load(f)

    print("Loading model ...")
    with open(os.path.join(args.model_path, "args"), "rb") as f:
        margs = pickle.load(f)
    n_days = len(data["train"])
    model, margs = load_conformer(args.model_path, n_days, args.device)

    print("Extracting logits (one time) ...")
    utterances, is_conf = extract_logits(
        model, margs, args.partition, data, args.device
    )
    print(f"  {len(utterances)} utterances\n")

    # ---- parameter grid ----
    if args.quick:
        lm_weights = [0.0, 0.5, 1.0, 2.0, 3.23]
        word_scores = [-0.26, 0.0]
        beam_sizes = [500]
        blank_penalties = [0.0]
    else:
        lm_weights = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.23, 5.0]
        word_scores = [-1.0, -0.5, -0.26, 0.0, 0.5, 1.0]
        beam_sizes = [500]
        blank_penalties = [0.0, 1.0, 2.0]

    print(
        f"{'lm_weight':>10} {'word_score':>11} {'beam':>6} {'blank_pen':>10} {'WER':>8} {'edit':>6} {'ref':>6}  {'time':>6}"
    )
    print("-" * 80)

    best_wer = float("inf")
    best_params = {}

    for lm_w, ws, bs, bp in itertools.product(
        lm_weights, word_scores, beam_sizes, blank_penalties
    ):
        try:
            decoder = CTCBeamSearchDecoder(
                lm_dir=args.lm_dir,
                lm_weight=lm_w,
                word_score=ws,
                beam_size=bs,
                blank_penalty=bp,
            )
        except Exception as e:
            print(f"  SKIP lm_w={lm_w} ws={ws} bs={bs} bp={bp}: {e}")
            continue

        t0 = time.time()
        wer, ed, nref, _ = decode_and_score(utterances, is_conf, decoder)
        elapsed = time.time() - t0

        marker = ""
        if wer < best_wer:
            best_wer = wer
            best_params = {
                "lm_weight": lm_w,
                "word_score": ws,
                "beam_size": bs,
                "blank_penalty": bp,
            }
            marker = " <-- BEST"

        print(
            f"{lm_w:10.2f} {ws:11.2f} {bs:6d} {bp:10.1f} "
            f"{100 * wer:7.2f}% {ed:6d} {nref:6d}  {elapsed:5.1f}s{marker}"
        )

    # ---- print best ----
    print()
    print("=" * 80)
    print(f"  BEST WER: {100 * best_wer:.2f}%")
    print(f"  Params:   {best_params}")
    print("=" * 80)

    # ---- show samples with best params ----
    print("\nDecoding with best params for sample output ...")
    decoder = CTCBeamSearchDecoder(
        lm_dir=args.lm_dir,
        lm_weight=best_params["lm_weight"],
        word_score=best_params["word_score"],
        beam_size=best_params["beam_size"],
        blank_penalty=best_params["blank_penalty"],
    )
    _, _, _, preds = decode_and_score(utterances, is_conf, decoder)
    print("\nSamples:")
    for i in range(min(20, len(preds))):
        print(f"  [{i}] ref:  {remove_punctuation(utterances[i]['transcription'])}")
        print(f"       pred: {remove_punctuation(preds[i])}")

    # ---- print the command to use ----
    print(f"\nTo reproduce:")
    print(f"  uv run python scripts/eval_posterior_constrained.py \\")
    print(f"      --model_path {args.model_path} \\")
    print(f"      --dataset_path {args.dataset_path} \\")
    print(f"      --partition {args.partition} \\")
    print(f"      --lm_dir {args.lm_dir} \\")
    print(f"      --lm_weight {best_params['lm_weight']} \\")
    print(f"      --word_score {best_params['word_score']} \\")
    print(f"      --beam_size {best_params['beam_size']} \\")
    print(f"      --blank_penalty {best_params['blank_penalty']}")


if __name__ == "__main__":
    main()
