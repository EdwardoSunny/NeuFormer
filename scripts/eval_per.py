"""
Evaluate Phoneme Error Rate (PER) of the Conformer CTC model.

Greedy-decodes the CTC output (argmax → collapse repeats → remove blanks)
and compares against the ground-truth phoneme sequences from the dataset.
This matches the PER computation used during training.

Usage
-----
    uv run python scripts/eval_per.py \
        --model_path logs/speech_logs/conformer_v2 \
        --dataset_path ptDecoder_ctc \
        --partition test
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edit_distance import SequenceMatcher

from neural_decoder.dataset import SpeechDataset
from neural_decoder.model import GRUDecoder
from neural_decoder.phoneme_table import ID_TO_PHONE, BLANK_IDX, SIL_IDX, PHONE_DEF_SIL
from neural_decoder.transformer_ctc import NeuralTransformerCTCModel


# ------------------------------------------------------------------
# Model loading (same as eval_posterior_constrained.py)
# ------------------------------------------------------------------
def load_conformer(model_dir: str, n_days: int = 24, device: str = "cuda"):
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
        use_depthwise_frontend=args.get("use_depthwise_frontend", False),
        depthwise_hidden_dim=args.get("depthwise_hidden_dim", 1024),
        decoder_layers=args.get("decoder_layers", 0),
        decoder_ff_dim=args.get("decoder_ff_dim", 0),
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


def load_gru(model_dir: str, n_days: int = 24, device: str = "cuda"):
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
# Collate
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate Phoneme Error Rate (PER)")
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
    ap.add_argument(
        "--show_samples",
        type=int,
        default=20,
        help="Number of sample utterances to print (default: 20)",
    )
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

    device = args.device

    # ---- evaluate PER ----
    total_edit_distance = 0
    total_seq_length = 0
    samples = []  # collect (true_phones, pred_phones) for display

    t0 = time.time()
    for day_idx in tqdm(range(len(data[args.partition])), desc="Days", unit="day"):
        day = data[args.partition][day_idx]
        loader = DataLoader(
            SpeechDataset([day]),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        )

        for X, y, X_len, y_len, _ in loader:
            X, y, X_len, y_len = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
            )
            day_t = torch.tensor([day_idx], dtype=torch.int64, device=device)

            with torch.no_grad():
                if is_conformer:
                    out = model(X, day_t, X_len)
                    pred, adjustedLens = out[0], out[1]
                    # pred is [T, B, C] (log-probs)
                else:
                    logits = model.forward(X, day_t)
                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    pred = logits.log_softmax(2).permute(1, 0, 2)

            for iterIdx in range(pred.shape[1]):
                # Greedy decode: argmax → collapse repeats → remove blanks
                decodedSeq = torch.argmax(
                    pred[0 : adjustedLens[iterIdx], iterIdx, :],
                    dim=-1,
                )
                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                decodedSeq = decodedSeq.cpu().numpy()
                decodedSeq = np.array([i for i in decodedSeq if i != BLANK_IDX])

                trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

                matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
                ed = matcher.distance()
                total_edit_distance += ed
                total_seq_length += len(trueSeq)

                # Collect sample for display
                if len(samples) < args.show_samples:
                    true_phones = [ID_TO_PHONE.get(int(i), f"?{i}") for i in trueSeq]
                    pred_phones = [ID_TO_PHONE.get(int(i), f"?{i}") for i in decodedSeq]
                    samples.append((true_phones, pred_phones, ed, len(trueSeq)))

    elapsed = time.time() - t0
    per = total_edit_distance / max(total_seq_length, 1)

    # ---- report ----
    print(f"\n{'=' * 50}")
    print(f"  Partition:        {args.partition}")
    print(f"  Total phonemes:   {total_seq_length}")
    print(f"  Edit distance:    {total_edit_distance}")
    print(f"  PER:              {100 * per:.2f}%")
    print(f"  Eval time:        {elapsed:.1f}s")
    print(f"{'=' * 50}")

    print(f"\nSamples (first {len(samples)}):")
    for i, (true_ph, pred_ph, ed, n_ref) in enumerate(samples):
        sent_per = ed / max(n_ref, 1)
        print(f"  [{i}] PER: {100 * sent_per:.1f}%  ({ed}/{n_ref})")
        print(f"       ref:  {' '.join(true_ph)}")
        print(f"       pred: {' '.join(pred_ph)}")


if __name__ == "__main__":
    main()
