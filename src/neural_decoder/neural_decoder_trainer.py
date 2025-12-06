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
import wandb

from .model import GRUDecoder
from .transformer_ctc import NeuralTransformerCTCModel
from .dataset import SpeechDataset


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

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

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

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
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

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
        mode=args.get("wandb_mode", "online"),  # Can be "online", "offline", or "disabled"
    )

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
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
    wandb.log({
        "model/total_parameters": num_params,
        "model/trainable_parameters": num_trainable_params,
    })
    print(f"Model has {num_params:,} parameters ({num_trainable_params:,} trainable)")

    blank_idx = 0
    n_classes = args["nClasses"] + 1  # +1 for blank

    # Label smoothing for better generalization (Conformer only)
    label_smoothing = args.get("label_smoothing", 0.0)
    if label_smoothing > 0:
        loss_ctc = torch.nn.CTCLoss(blank=blank_idx, reduction="none", zero_infinity=True)
    else:
        loss_ctc = torch.nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)

    # Optimizer and scheduler
    if args.get("optimizer", "adam") == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=1e-6,  # Increased from 1e-8 for stability with mixed precision
            weight_decay=args.get("weight_decay", args.get("l2_decay", 0)),
        )
        warmup_steps = int(args.get("warmup_steps", 0))
        total_steps = args["nBatch"]

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

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
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Compute prediction error
        inter_log_probs = None
        if args.get("model_type", "gru_baseline") == "transformer_ctc":
            log_probs, out_lens, inter_log_probs = model(X, dayIdx, X_len)
        else:
            pred = model.forward(X, dayIdx)
            out_lens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
            log_probs = pred.log_softmax(2).permute(1, 0, 2)

        # Main CTC loss
        loss = loss_ctc(
            log_probs,
            y,
            out_lens,
            y_len,
        )

        # Add intermediate CTC loss if available
        interctc_weight = args.get("interctc_weight", 0.3)
        if inter_log_probs is not None:
            inter_loss = loss_ctc(
                inter_log_probs,
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
            uniform_dist = torch.full_like(log_probs, -math.log(n_classes))
            kl_div = torch.nn.functional.kl_div(log_probs, uniform_dist, reduction='batchmean', log_target=True)
            main_loss = (1 - label_smoothing) * ctc_loss + label_smoothing * kl_div
        else:
            main_loss = torch.sum(loss)

        # Combine main loss with intermediate CTC loss
        if inter_log_probs is not None:
            loss = (1.0 - interctc_weight) * main_loss + interctc_weight * inter_loss
        else:
            loss = main_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (especially important for Conformer)
        grad_norm = None
        if args.get("model_type", "gru_baseline") == "transformer_ctc":
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Log training metrics every step
        current_lr = optimizer.param_groups[0]['lr']
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
            log_dict["train/main_loss"] = main_loss.item()
        wandb.log(log_dict, step=batch)

        # Eval
        if batch % 100 == 0:
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
                        pred, adjustedLens, _ = model(X, testDayIdx, X_len)
                        # pred is [T, B, C], ignore intermediate output during eval
                    else:
                        logits = model.forward(X, testDayIdx)
                        adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
                        pred = logits.log_softmax(2).permute(1, 0, 2)

                    loss = loss_ctc(
                        pred,
                        y,
                        adjustedLens,
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

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
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {time_per_batch:>7.3f}"
                )
                startTime = time.time()

                # Log evaluation metrics
                eval_log_dict = {
                    "eval/loss": avgDayLoss,
                    "eval/cer": cer,
                    "eval/time_per_batch": time_per_batch,
                    "eval/edit_distance": total_edit_distance,
                    "eval/sequence_length": total_seq_length,
                }
                wandb.log(eval_log_dict, step=batch)

            # Save best model
            is_best = False
            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                is_best = True
                wandb.log({"eval/best_cer": cer}, step=batch)
                print(f"  → New best model saved! CER: {cer:.6f}")

            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)

    # Log final summary
    final_cer = testCER[-1] if len(testCER) > 0 else float('inf')
    best_cer = np.min(testCER) if len(testCER) > 0 else float('inf')
    wandb.log({
        "summary/final_cer": final_cer,
        "summary/best_cer": best_cer,
        "summary/final_loss": testLoss[-1] if len(testLoss) > 0 else float('inf'),
        "summary/best_loss": np.min(testLoss) if len(testLoss) > 0 else float('inf'),
    })

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final CER: {final_cer:.6f}")
    print(f"Best CER: {best_cer:.6f}")
    print(f"{'='*60}\n")

    # Finish wandb run
    wandb.finish()


def loadModel(modelDir, nInputLayers=24, device="cuda"):
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


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()