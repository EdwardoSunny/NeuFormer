import math
import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from .model import GRUDecoder
from .transformer_ctc import NeuralTransformerCTCModel
from .dataset import SpeechDataset, N_CHARS


def getDatasetLoaders(
    datasetName,
    batchSize,
    load_chars=False,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        if load_chars:
            X, y, X_lens, y_lens, days, c, c_lens = zip(*batch)
        else:
            X, y, X_lens, y_lens, days = zip(*batch)

        X_padded = pad_sequence(list(X), batch_first=True, padding_value=0)
        y_padded = pad_sequence(list(y), batch_first=True, padding_value=0)

        result = (
            X_padded,
            y_padded,
            torch.stack(list(X_lens)),
            torch.stack(list(y_lens)),
            torch.stack(list(days)),
        )

        if load_chars:
            c_padded = pad_sequence(list(c), batch_first=True, padding_value=0)
            result = result + (c_padded, torch.stack(list(c_lens)))

        return result

    train_ds = SpeechDataset(loadedData["train"], transform=None, load_chars=load_chars)
    test_ds = SpeechDataset(loadedData["test"], load_chars=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda batch: _padding_simple(batch),
    )

    return train_loader, test_loader, loadedData


def _padding_simple(batch):
    """Collate for test loader (never loads chars)."""
    X, y, X_lens, y_lens, days = zip(*batch)
    X_padded = pad_sequence(list(X), batch_first=True, padding_value=0)
    y_padded = pad_sequence(list(y), batch_first=True, padding_value=0)
    return (
        X_padded,
        y_padded,
        torch.stack(list(X_lens)),
        torch.stack(list(y_lens)),
        torch.stack(list(days)),
    )


def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    # Initialize wandb
    wandb.init(
        project=args.get("wandb_project", "neural-speech-decoder"),
        name=args.get("wandb_run_name", os.path.basename(args["outputDir"])),
        config=args,
        mode=args.get(
            "wandb_mode", "online"
        ),  # Can be "online", "offline", or "disabled"
    )

    # Dual-task CTC config
    use_char_ctc = args.get("use_char_ctc", False)
    char_ctc_weight = args.get("char_ctc_weight", 0.3)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
        load_chars=use_char_ctc,
    )

    # Create model based on type
    if args.get("model_type", "gru_baseline") == "transformer_ctc":
        model = NeuralTransformerCTCModel(
            n_channels=args["nInputFeatures"],
            n_classes=args["nClasses"] + 1,  # +1 for CTC blank
            n_days=len(loadedData["train"]),
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
            use_spec_augment=args.get("use_spec_augment", True),
            spec_augment_freq_mask=args.get("spec_augment_freq_mask", 100),
            spec_augment_time_mask=args.get("spec_augment_time_mask", 40),
            drop_path_prob=args.get("drop_path_prob", 0.1),
            autoencoder_residual=args.get("autoencoder_residual", False),
            use_rope=args.get("use_rope", False),
            n_chars=N_CHARS if use_char_ctc else 0,
            use_depthwise_frontend=args.get("use_depthwise_frontend", False),
            depthwise_hidden_dim=args.get("depthwise_hidden_dim", 1024),
            decoder_layers=args.get("decoder_layers", 0),
            decoder_ff_dim=args.get("decoder_ff_dim", 0),
            device=device,
        ).to(device)
    else:
        model = GRUDecoder(
            neural_dim=args["nInputFeatures"],
            n_classes=args["nClasses"],
            hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"],
            nDays=len(loadedData["train"]),
            dropout=args["dropout"],
            device=device,
            strideLen=args["strideLen"],
            kernelLen=args["kernelLen"],
            gaussianSmoothWidth=args["gaussianSmoothWidth"],
            bidirectional=args["bidirectional"],
        ).to(device)

    # Watch model gradients and parameters
    wandb.watch(model, log="all", log_freq=100)

    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log(
        {
            "model/total_parameters": num_params,
            "model/trainable_parameters": num_trainable_params,
        }
    )
    print(f"Model has {num_params:,} parameters ({num_trainable_params:,} trainable)")

    blank_idx = 0
    n_classes = args["nClasses"] + 1  # +1 for blank

    # Label smoothing for better generalization (Conformer only)
    label_smoothing = args.get("label_smoothing", 0.0)
    if label_smoothing > 0:
        loss_ctc = torch.nn.CTCLoss(
            blank=blank_idx, reduction="none", zero_infinity=True
        )
    else:
        loss_ctc = torch.nn.CTCLoss(
            blank=blank_idx, reduction="mean", zero_infinity=True
        )

    # Character CTC loss (always uses reduction="none" for averaging)
    if use_char_ctc:
        loss_char_ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)

    # v4 augmentation config
    mult_noise_sd = args.get("mult_noise_sd", 0.0)
    temporal_shift_max = args.get("temporal_shift_max", 0)

    # Optimizer and scheduler
    if args.get("optimizer", "adam") == "adamw":
        beta2 = args.get("beta2", 0.999)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, beta2),
            eps=1e-6,  # Increased from 1e-8 for stability with mixed precision
            weight_decay=args.get("weight_decay", args.get("l2_decay", 0)),
        )
        warmup_steps = int(args.get("warmup_steps", 0))
        total_steps = args["nBatch"]
        # Cosine schedule decays to lrEnd (not zero) for stable late training
        lr_min_ratio = args.get("lrEnd", 0.0) / max(args["lrStart"], 1e-10)

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=args["l2_decay"],
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args["lrEnd"] / args["lrStart"],
            total_iters=args["nBatch"],
        )

    # --train--
    # EMA (Exponential Moving Average) of model weights for better generalization
    use_ema = args.get("use_ema", False)
    ema_decay = args.get("ema_decay", 0.999)
    ema_state = None
    if use_ema:
        ema_state = {k: v.clone() for k, v in model.state_dict().items()}

    # AMP (Automatic Mixed Precision) for faster training
    use_amp = args.get("use_amp", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    testLoss = []
    testCER = []
    startTime = time.time()

    # Fix: use a proper iterator that cycles through the full dataset
    # instead of next(iter(trainLoader)) which re-creates iterator every batch
    train_iter = iter(trainLoader)

    pbar = tqdm(range(args["nBatch"]), desc="Training", unit="step")
    for batch in pbar:
        model.train()

        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(trainLoader)
            batch_data = next(train_iter)

        if use_char_ctc:
            X, y, X_len, y_len, dayIdx, char_y, char_y_len = batch_data
            char_y = char_y.to(device)
            char_y_len = char_y_len.to(device)
        else:
            X, y, X_len, y_len, dayIdx = batch_data

        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # -- Data augmentation --

        # Multiplicative noise (v4): simulates electrode gain changes
        if mult_noise_sd > 0:
            X = X * (1.0 + torch.randn(X.shape, device=device) * mult_noise_sd)

        # Additive white noise
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        # Constant offset per channel (simulates DC drift)
        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Random temporal shift (v4): remove 0..N timesteps from start
        if temporal_shift_max > 0:
            shift = torch.randint(0, temporal_shift_max + 1, (1,)).item()
            if shift > 0:
                X = X[:, shift:, :]
                X_len = X_len - shift
                X_len = torch.clamp(X_len, min=1)

        # Compute prediction error (with optional AMP)
        inter_log_probs = None
        char_log_probs = None
        with torch.amp.autocast("cuda", enabled=use_amp):
            if args.get("model_type", "gru_baseline") == "transformer_ctc":
                log_probs, out_lens, inter_log_probs, char_log_probs = model(
                    X, dayIdx, X_len
                )
            else:
                pred = model.forward(X, dayIdx)
                out_lens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
                log_probs = pred.log_softmax(2).permute(1, 0, 2)

            # CTC loss needs float32 log_probs
            log_probs_f32 = log_probs.float()

            # Main CTC loss (phoneme)
            loss = loss_ctc(
                log_probs_f32,
                y,
                out_lens,
                y_len,
            )

            # Add intermediate CTC loss if available
            interctc_weight = args.get("interctc_weight", 0.3)
            if inter_log_probs is not None:
                inter_log_probs_f32 = inter_log_probs.float()
                inter_loss = loss_ctc(
                    inter_log_probs_f32,
                    y,
                    out_lens,
                    y_len,
                )
                if label_smoothing > 0:
                    inter_loss = torch.mean(inter_loss)
                else:
                    inter_loss = torch.sum(inter_loss)

            # Apply label smoothing and/or InterCTC
            if label_smoothing > 0:
                ctc_loss = torch.mean(loss)
                # Uniform distribution for label smoothing
                uniform_dist = torch.full_like(log_probs_f32, -math.log(n_classes))
                kl_div = torch.nn.functional.kl_div(
                    log_probs_f32, uniform_dist, reduction="batchmean", log_target=True
                )
                main_loss = (1 - label_smoothing) * ctc_loss + label_smoothing * kl_div
            else:
                main_loss = torch.sum(loss)

            # Combine main loss with intermediate CTC loss
            if inter_log_probs is not None:
                phone_loss = (
                    1.0 - interctc_weight
                ) * main_loss + interctc_weight * inter_loss
            else:
                phone_loss = main_loss

            # Character CTC loss (dual-task, v4)
            total_loss = phone_loss
            if use_char_ctc and char_log_probs is not None:
                char_log_probs_f32 = char_log_probs.float()
                char_loss_raw = loss_char_ctc(
                    char_log_probs_f32,
                    char_y,
                    out_lens,
                    char_y_len,
                )
                char_loss = torch.mean(char_loss_raw)
                total_loss = (
                    1.0 - char_ctc_weight
                ) * phone_loss + char_ctc_weight * char_loss
            else:
                char_loss = None

            loss = total_loss

        # Backpropagation with AMP scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping (especially important for Conformer)
        grad_norm = None
        if args.get("model_type", "gru_baseline") == "transformer_ctc":
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        if batch > 0 or not use_amp:
            scheduler.step()

        # Update EMA
        if use_ema and ema_state is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_state[k].mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)

        # Build log dict for this step (train metrics always included)
        current_lr = optimizer.param_groups[0]["lr"]
        log_dict = {
            "train/loss": loss.item(),
            "train/learning_rate": current_lr,
            "train/batch": batch,
        }
        if grad_norm is not None:
            log_dict["train/grad_norm"] = grad_norm.item()
        if label_smoothing > 0:
            log_dict["train/ctc_loss"] = ctc_loss.item()
            log_dict["train/kl_loss"] = kl_div.item()
        if inter_log_probs is not None:
            log_dict["train/inter_ctc_loss"] = inter_loss.item()
            log_dict["train/phone_loss"] = phone_loss.item()
        if char_loss is not None:
            log_dict["train/char_ctc_loss"] = char_loss.item()

        # Eval
        if batch % 100 == 0:
            # Swap in EMA weights for evaluation if enabled
            if use_ema and ema_state is not None:
                original_state = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state)

            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    if args.get("model_type", "gru_baseline") == "transformer_ctc":
                        pred, adjustedLens, _, _ = model(X, testDayIdx, X_len)
                        # pred is [T, B, C], ignore intermediate/char output during eval
                    else:
                        logits = model.forward(X, testDayIdx)
                        adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                            torch.int32
                        )
                        pred = logits.log_softmax(2).permute(1, 0, 2)

                    eval_loss = loss_ctc(
                        pred,
                        y,
                        adjustedLens,
                        y_len,
                    )
                    eval_loss = torch.sum(eval_loss)
                    allLoss.append(eval_loss.cpu().detach().numpy())

                    # Decode predictions - both models output [T, B, C] at this point
                    for iterIdx in range(pred.shape[1]):  # Iterate over batch dimension
                        decodedSeq = torch.argmax(
                            pred[0 : adjustedLens[iterIdx], iterIdx, :],
                            dim=-1,
                        )
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                time_per_batch = (endTime - startTime) / 100
                tqdm.write(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {time_per_batch:>7.3f}"
                )
                pbar.set_postfix(
                    loss=f"{avgDayLoss:.2f}", CER=f"{cer:.4f}", lr=f"{current_lr:.1e}"
                )
                startTime = time.time()

                # Add eval metrics to the same log dict
                log_dict.update(
                    {
                        "eval/loss": avgDayLoss,
                        "eval/cer": cer,
                        "eval/time_per_batch": time_per_batch,
                        "eval/edit_distance": total_edit_distance,
                        "eval/sequence_length": total_seq_length,
                    }
                )

            # Save best model
            is_best = False
            if len(testCER) == 0 or cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                is_best = True
                log_dict["eval/best_cer"] = cer
                tqdm.write(f"  -> New best model saved! CER: {cer:.6f}")

            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)

            # Restore original (non-EMA) weights for continued training
            if use_ema and ema_state is not None:
                model.load_state_dict(original_state)

        # Single wandb.log per step (train + eval merged)
        wandb.log(log_dict, step=batch)

    # Log final summary
    final_cer = testCER[-1] if len(testCER) > 0 else float("inf")
    best_cer = np.min(testCER) if len(testCER) > 0 else float("inf")
    wandb.log(
        {
            "summary/final_cer": final_cer,
            "summary/best_cer": best_cer,
            "summary/final_loss": testLoss[-1] if len(testLoss) > 0 else float("inf"),
            "summary/best_loss": np.min(testLoss)
            if len(testLoss) > 0
            else float("inf"),
        }
    )

    print(f"\n{'=' * 60}")
    print(f"Training completed!")
    print(f"Final CER: {final_cer:.6f}")
    print(f"Best CER: {best_cer:.6f}")
    print(f"{'=' * 60}\n")

    # Finish wandb run
    wandb.finish()


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    """Load a trained GRU model from a saved directory."""
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


def loadConformerModel(modelDir, nInputLayers=24, device="cuda"):
    """Load a trained Conformer model from a saved directory."""
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = NeuralTransformerCTCModel(
        n_channels=args["nInputFeatures"],
        n_classes=args["nClasses"] + 1,  # +1 for CTC blank
        n_days=nInputLayers,
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
        use_spec_augment=False,  # Disabled during inference
        drop_path_prob=0.0,  # Disabled during inference
        autoencoder_residual=args.get("autoencoder_residual", False),
        use_rope=args.get("use_rope", False),
        n_chars=0,  # Char head not needed at inference
        use_depthwise_frontend=args.get("use_depthwise_frontend", False),
        depthwise_hidden_dim=args.get("depthwise_hidden_dim", 1024),
        decoder_layers=args.get("decoder_layers", 0),
        decoder_ff_dim=args.get("decoder_ff_dim", 0),
        device=device,
    ).to(device)

    # Load with strict=False to allow missing char_output keys at inference
    model.load_state_dict(
        torch.load(modelWeightPath, map_location=device), strict=False
    )
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)


if __name__ == "__main__":
    main()
