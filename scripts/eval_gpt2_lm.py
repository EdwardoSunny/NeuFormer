"""
Evaluate Conformer + CTC Beam Search with GPT-2 Neural LM Rescoring.

Uses GPT-2 as the language model in the CTC beam search decoder instead
of KenLM n-grams.  This provides much stronger contextual rescoring.

Note: GPT-2 LM rescoring is ~100x slower than KenLM, so use a smaller
beam size (50-100) compared to KenLM (1500).

Usage
-----
    uv run python scripts/eval_gpt2_lm.py \\
        --model_path logs/speech_logs/conformer_v2 \\
        --dataset_path ptDecoder_ctc \\
        --partition test \\
        --lm_dir lm_dir \\
        --lm_weight 2.0 \\
        --beam_size 50
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
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_decoder.dataset import SpeechDataset
from neural_decoder.evaluation import compute_wer, remove_punctuation
from neural_decoder.model import GRUDecoder
from neural_decoder.ngram_decoder import (
    CTCBeamSearchDecoder,
    GPT2CTCDecoderLM,
    TOKENS,
)
from neural_decoder.transformer_ctc import NeuralTransformerCTCModel


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
    model.load_state_dict(state, strict=False)
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


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate Conformer + CTC beam search with GPT-2 LM"
    )
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--partition", default="test", choices=["test", "competition"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lm_dir", required=True)
    ap.add_argument("--gpt2_model", default="gpt2", help="HuggingFace GPT-2 model ID")
    ap.add_argument("--lm_weight", type=float, default=2.0)
    ap.add_argument("--word_score", type=float, default=0.0)
    ap.add_argument("--beam_size", type=int, default=50)
    ap.add_argument("--beam_threshold", type=float, default=50.0)
    ap.add_argument(
        "--max_utts",
        type=int,
        default=0,
        help="Max utterances to decode (0=all, useful for quick sweeps)",
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    # Load data + model
    print("Loading dataset ...")
    with open(args.dataset_path, "rb") as f:
        data = pickle.load(f)

    print("Loading model ...")
    with open(os.path.join(args.model_path, "args"), "rb") as f:
        margs = pickle.load(f)
    n_days = len(data["train"])
    model, margs = load_conformer(args.model_path, n_days, args.device)

    utterances, is_conf = extract_logits(
        model, margs, args.partition, data, args.device
    )
    print(f"  {len(utterances)} utterances")

    # Build GPT-2 LM
    print(f"Loading GPT-2 LM ({args.gpt2_model}) ...")
    gpt2_lm = GPT2CTCDecoderLM(model_id=args.gpt2_model, device=args.device)

    lex_path = os.path.join(args.lm_dir, "lexicon.txt")
    tok_path = os.path.join(args.lm_dir, "tokens.txt")
    lm_obj = gpt2_lm.build_lm(lex_path, tok_path)

    # Build decoder with GPT-2 LM
    decoder = CTCBeamSearchDecoder(
        lm_dir=args.lm_dir,
        lm=lm_obj,
        lm_weight=args.lm_weight,
        word_score=args.word_score,
        beam_size=args.beam_size,
        beam_threshold=args.beam_threshold,
    )
    print(
        f"Using GPT-2 CTC beam search decoder "
        f"(beam={args.beam_size}, lm_weight={args.lm_weight})"
    )

    # Decode
    if args.max_utts > 0:
        utterances = utterances[: args.max_utts]
        print(f"  (limited to {args.max_utts} utterances for quick sweep)")
    preds, refs = [], []
    t0 = time.time()
    pbar = tqdm(utterances, desc="Decoding", unit="utt")
    for i, utt in enumerate(pbar):
        pred = decoder.decode_text(utt["logits"], is_log_probs=is_conf)
        preds.append(pred)
        refs.append(utt["transcription"])
        # Show running WER in progress bar
        if any(r.strip() for r in refs):
            running_wer, _, _ = compute_wer(preds, refs)
            pbar.set_postfix(WER=f"{100 * running_wer:.1f}%")
        if i < 5:
            tqdm.write(f"  [{i}] ref:  {remove_punctuation(refs[-1])}")
            tqdm.write(f"       pred: {remove_punctuation(preds[-1])}")
    elapsed = time.time() - t0

    # WER
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


if __name__ == "__main__":
    main()
