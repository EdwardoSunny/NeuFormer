"""
Evaluate Conformer + CTC Beam Search Decoder with KenLM n-gram LM.

Uses ``torchaudio.models.decoder.ctc_decoder`` for lexicon-constrained
beam search decoding.  Supports greedy fallback when no LM is provided.

Usage
-----
::

    # greedy (no LM):
    python scripts/eval_posterior_constrained.py \\
        --model_path logs/conformer \\
        --dataset_path ptDecoder_ctc

    # with KenLM n-gram beam search:
    python scripts/eval_posterior_constrained.py \\
        --model_path logs/conformer \\
        --dataset_path ptDecoder_ctc \\
        --lm_dir language_model/pretrained_language_models/languageModel \\
        --lm_weight 3.23 \\
        --beam_size 1500

    # with custom beam parameters:
    python scripts/eval_posterior_constrained.py \\
        --model_path logs/conformer \\
        --dataset_path ptDecoder_ctc \\
        --lm_dir language_model/pretrained_language_models/languageModel \\
        --lm_weight 3.23 \\
        --word_score -0.26 \\
        --beam_size 1500 \\
        --beam_threshold 50.0 \\
        --blank_penalty 0.0 \\
        --nbest 3
"""

import argparse
import csv
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
from neural_decoder.ngram_decoder import (
    CTCBeamSearchDecoder,
    GreedyCTCDecoder,
    TOKENS,
)
from neural_decoder.transformer_ctc import NeuralTransformerCTCModel


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
def load_conformer(model_dir: str, n_days: int = 24, device: str = "cuda"):
    """Load a trained Conformer checkpoint."""
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


def load_gru(model_dir: str, n_days: int = 24, device: str = "cuda"):
    """Load a trained GRU baseline checkpoint."""
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
    state = torch.load(
        os.path.join(model_dir, "modelWeights"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model, args


# ------------------------------------------------------------------
# Logit extraction
# ------------------------------------------------------------------
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
    """Extract per-utterance CTC log-probabilities from the model."""
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
                    out = model(X, day_t, X_len)
                    lp = out[0][:, 0, :].cpu().numpy()
                    ol = out[1][0].cpu().item()
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate Conformer + CTC beam search decoder with KenLM"
    )
    ap.add_argument("--model_path", required=True, help="Path to model checkpoint dir")
    ap.add_argument(
        "--dataset_path", required=True, help="Path to ptDecoder_ctc pickle"
    )
    ap.add_argument(
        "--partition",
        default="test",
        choices=["test", "competition"],
    )
    ap.add_argument("--device", default="cuda")

    # LM / beam search decoder args
    ap.add_argument(
        "--lm_dir",
        default=None,
        help="Dir with lexicon.txt, tokens.txt, lm.arpa/bin. "
        "Omit for greedy decoding (no LM).",
    )
    ap.add_argument(
        "--lm_weight",
        type=float,
        default=3.23,
        help="Language model weight (default: 3.23)",
    )
    ap.add_argument(
        "--word_score",
        type=float,
        default=-0.26,
        help="Per-word insertion bonus (default: -0.26)",
    )
    ap.add_argument(
        "--sil_score",
        type=float,
        default=0.0,
        help="Silence appearance bonus (default: 0.0)",
    )
    ap.add_argument(
        "--beam_size", type=int, default=1500, help="Beam width (default: 1500)"
    )
    ap.add_argument(
        "--beam_size_token",
        type=int,
        default=None,
        help="Tokens to consider per step (default: all)",
    )
    ap.add_argument(
        "--beam_threshold",
        type=float,
        default=50.0,
        help="Beam pruning threshold (default: 50.0)",
    )
    ap.add_argument(
        "--blank_penalty",
        type=float,
        default=0.0,
        help="Penalty subtracted from blank log-prob (default: 0.0)",
    )
    ap.add_argument(
        "--nbest", type=int, default=1, help="Number of n-best hypotheses (default: 1)"
    )

    ap.add_argument("--output", default=None, help="CSV output path")
    args = ap.parse_args()

    # ---- load data + model ----
    print("Loading dataset ...")
    with open(args.dataset_path, "rb") as f:
        data = pickle.load(f)

    print("Loading model ...")
    with open(os.path.join(args.model_path, "args"), "rb") as f:
        margs = pickle.load(f)
    is_conformer = margs.get("model_type", "gru_baseline") == "transformer_ctc"
    n_days = len(data["train"])

    if is_conformer:
        model, margs = load_conformer(args.model_path, n_days, args.device)
    else:
        model, margs = load_gru(args.model_path, n_days, args.device)

    utterances, is_conf = extract_logits(
        model, margs, args.partition, data, args.device
    )
    print(f"  {len(utterances)} utterances")

    # ---- build decoder ----
    decoder = None  # set below when using beam search
    if args.lm_dir:
        decoder = CTCBeamSearchDecoder(
            lm_dir=args.lm_dir,
            lm_weight=args.lm_weight,
            word_score=args.word_score,
            sil_score=args.sil_score,
            beam_size=args.beam_size,
            beam_size_token=args.beam_size_token,
            beam_threshold=args.beam_threshold,
            nbest=args.nbest,
            blank_penalty=args.blank_penalty,
        )
        print(
            f"Using CTC beam search decoder "
            f"(lm_dir={args.lm_dir}, beam={args.beam_size}, "
            f"lm_weight={args.lm_weight})"
        )

        def decode_fn(lp):
            return decoder.decode_text(lp, is_log_probs=is_conf)

    else:
        greedy = GreedyCTCDecoder(tokens=TOKENS)
        print("Using greedy decoder (no LM)")

        def decode_fn(lp):
            em = torch.from_numpy(np.ascontiguousarray(lp, dtype=np.float32))
            words = greedy(em)
            return " ".join(words)

    # ---- decode ----
    preds, refs = [], []
    t0 = time.time()
    for utt in tqdm(utterances, desc="Decoding", unit="utt"):
        preds.append(decode_fn(utt["logits"]))
        refs.append(utt["transcription"])
    elapsed = time.time() - t0

    # ---- print n-best if requested ----
    if decoder is not None and args.nbest > 1:
        print(f"\nShowing top-{args.nbest} hypotheses for first 5 utterances:")
        for i in range(min(5, len(utterances))):
            hyps = decoder.decode_nbest(
                utterances[i]["logits"], n=args.nbest, is_log_probs=is_conf
            )
            print(f"  [{i}] ref: {remove_punctuation(refs[i])}")
            for j, h in enumerate(hyps):
                txt = " ".join(h.words).strip()
                print(f"       [{j}] {txt}  (score: {h.score:.2f})")

    # ---- WER ----
    has_refs = any(r.strip() for r in refs)
    if has_refs:
        wer, ed, nref = compute_wer(preds, refs)
        print(f"\n{'=' * 50}")
        print(f"  Partition:    {args.partition}")
        print(f"  Utterances:   {len(utterances)}")
        print(f"  Ref words:    {nref}")
        print(f"  Edit dist:    {ed}")
        print(f"  WER:          {100 * wer:.2f}%")
        print(f"  Decode time:  {elapsed:.1f}s")
        print(f"{'=' * 50}")
        print("\nSamples:")
        for i in range(min(20, len(preds))):
            print(f"  [{i}] ref:  {remove_punctuation(refs[i])}")
            print(f"       pred: {remove_punctuation(preds[i])}")
    else:
        print(f"\nNo references for {args.partition}. First 20 predictions:")
        for i in range(min(20, len(preds))):
            print(f"  [{i}] {preds[i]}")

    # ---- save ----
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            for i, p in enumerate(preds):
                w.writerow([i, p])
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
