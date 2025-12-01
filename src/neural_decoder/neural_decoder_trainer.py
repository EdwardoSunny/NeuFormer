import math
import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import wandb

from .model import GRUDecoder
from .transformer_ctc import NeuralTransformerCTCModel
from .dataset import SpeechDataset


def getDatasetLoaders(
    datasetName,
    batchSize,
    diphone_vocab=None,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        if diphone_vocab is not None:
            # Unpack with diphone labels
            X, y, X_lens, y_lens, days, y_diphone, y_diphone_lens = zip(*batch)
            X_padded = pad_sequence(X, batch_first=True, padding_value=0)
            y_padded = pad_sequence(y, batch_first=True, padding_value=0)
            y_diphone_padded = pad_sequence(y_diphone, batch_first=True, padding_value=0)

            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
                y_diphone_padded,
                torch.stack(y_diphone_lens),
            )
        else:
            # Original format without diphones
            X, y, X_lens, y_lens, days, _, _ = zip(*batch)
            X_padded = pad_sequence(X, batch_first=True, padding_value=0)
            y_padded = pad_sequence(y, batch_first=True, padding_value=0)

            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
            )

    train_ds = SpeechDataset(loadedData["train"], transform=None, diphone_vocab=diphone_vocab)
    test_ds = SpeechDataset(loadedData["test"], diphone_vocab=diphone_vocab)

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


def _compute_output_lengths(model, x_lens):
    if hasattr(model, "compute_output_lengths"):
        return model.compute_output_lengths(x_lens)
    else:
        return ((x_lens - model.kernelLen) / model.strideLen).to(torch.int32)


def _get_diphone_alpha(batch, total_batches, schedule="constant", constant_alpha=0.5):
    """
    Compute alpha weight for phoneme vs diphone loss.
    alpha = 1.0 means 100% phoneme loss
    alpha = 0.0 means 100% diphone loss

    For joint training, we use:
    loss = alpha * phoneme_loss + (1 - alpha) * diphone_loss
    """
    if schedule == "constant":
        # Simple 50/50 weighting
        return constant_alpha
    elif schedule == "scheduled":
        # Paper-style schedule:
        # First 20%: alpha = 0.3 (lean more on diphones)
        # Middle 60%: linearly ramp from 0.3 → 0.7
        # Last 20%: alpha = 0.8 (focus more on phonemes)
        progress = batch / total_batches
        if progress < 0.2:
            return 0.3
        elif progress < 0.8:
            # Linear ramp from 0.3 to 0.7 over middle 60%
            ramp_progress = (progress - 0.2) / 0.6
            return 0.3 + 0.4 * ramp_progress
        else:
            return 0.8
    else:
        return 0.5


def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    # Initialize wandb
    wandb.init(
        project="neural-seq-decoder",
        config=args,
        name=os.path.basename(args["outputDir"]),
    )

    # Load diphone vocabulary if diphone training is enabled
    diphone_vocab = None
    diphone_marginalization_matrix = None
    if args.get("use_diphone_head", False):
        from neural_decoder.diphone_utils import DiphoneVocabulary
        diphone_vocab_path = args.get("diphone_vocab_path", "/home/edward/neural_seq_decoder/diphone_vocab.pkl")
        print(f"Loading diphone vocabulary from {diphone_vocab_path}")
        diphone_vocab = DiphoneVocabulary.load(diphone_vocab_path)

        # Create marginalization matrix for proper DCoND implementation
        if args.get("use_diphone_marginalization", True):  # Default to True for new approach
            print("Creating marginalization matrix for DCoND...")
            phoneme_blank_id = args["nClasses"]  # Blank is the last class index
            diphone_marginalization_matrix = diphone_vocab.create_marginalization_matrix(
                num_phonemes=args["nClasses"] + 1,  # +1 for blank
                phoneme_blank_id=phoneme_blank_id
            )
            print(f"Marginalization matrix shape: {diphone_marginalization_matrix.shape}")
            print("✓ Will use proper DCoND approach (diphone -> phoneme marginalization)")

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
        diphone_vocab=diphone_vocab,
    )

    if args.get("model_type", "gru_baseline") == "advanced_multiscale":
        from neural_decoder.advanced_multiscale import AdvancedMultiScaleDecoder
        model = AdvancedMultiScaleDecoder(
            n_classes=args["nClasses"] + 1,  # +1 for CTC blank
            input_dim=args["nInputFeatures"],
            d_model=args.get("latent_dim", 512),
            encoder_layers=args.get("transformer_num_layers", 6),
            n_heads=args.get("transformer_n_heads", 8),
            dim_ff=args.get("transformer_dim_ff", 2048),
            dropout=args.get("transformer_dropout", 0.1),
            conv_kernel=args.get("conformer_conv_kernel", 31),
            n_days=len(loadedData["train"]),
            gaussian_smooth_width=args.get("gaussian_smooth_width", 0.0),
            use_phonetic_features=args.get("use_phonetic_features", True),
            use_contrastive=args.get("use_contrastive", True),
            phonetic_loss_weight=args.get("phonetic_loss_weight", 0.2),
            contrastive_loss_weight=args.get("contrastive_loss_weight", 0.1),
            device=device,
        ).to(device)
    elif args.get("model_type", "gru_baseline") == "multiscale_ctc":
        from neural_decoder.transformer_ctc import MultiScaleCTCDecoder
        model = MultiScaleCTCDecoder(
            n_classes=args["nClasses"] + 1,  # +1 for CTC blank
            input_dim=args["nInputFeatures"],
            d_model=args.get("latent_dim", 512),
            encoder_layers=args.get("transformer_num_layers", 6),
            n_heads=args.get("transformer_n_heads", 8),
            dim_ff=args.get("transformer_dim_ff", 2048),
            dropout=args.get("transformer_dropout", 0.1),
            conv_kernel=args.get("conformer_conv_kernel", 31),
            n_days=len(loadedData["train"]),
            gaussian_smooth_width=args.get("gaussian_smooth_width", 0.0),
            use_contrastive_pretraining=args.get("use_contrastive_pretraining", False),
            use_diphone_head=args.get("use_diphone_head", False),
            num_diphones=diphone_vocab.get_vocab_size() if diphone_vocab is not None else 1012,
            diphone_marginalization_matrix=diphone_marginalization_matrix,  # Step 1+2: Pass marginalization matrix
            use_multiscale_ctc=args.get("use_multiscale_ctc", False),  # Step 3: Enable multi-scale CTC heads
            device=device,
        ).to(device)
    elif args.get("model_type", "gru_baseline") == "transformer_ctc":
        model = NeuralTransformerCTCModel(
            n_channels=args["nInputFeatures"],
            n_classes=args["nClasses"] + 1,  # CTC blank already added in GRU; align dimensions
            n_days=len(loadedData["train"]),
            use_wavelets=args.get("use_wavelets", True),
            n_scales=args.get("wavelet_n_scales", 4),
            wavelet_window_size=args.get("wavelet_window_size", 10),
            use_pac_features=args.get("use_pac_features", False),
            frontend_dim=args.get("frontend_dim", 512),
            latent_dim=args.get("latent_dim", 256),
            autoencoder_hidden_dim=args.get("autoencoder_hidden_dim", 128),
            transformer_layers=args.get("transformer_num_layers", 4),
            transformer_heads=args.get("transformer_n_heads", 4),
            transformer_ff_dim=args.get("transformer_dim_ff", 1024),
            transformer_dropout=args.get("transformer_dropout", 0.1),
            temporal_kernel=args.get("temporal_kernel", 0),
            temporal_stride=args.get("temporal_stride", 1),
            gaussian_smooth_width=args.get("gaussian_smooth_width", 0.0),
            use_conformer=args.get("use_conformer", False),
            conformer_conv_kernel=args.get("conformer_conv_kernel", 31),
            use_relative_pe=args.get("use_relative_pe", True),
            intermediate_ctc_layers=args.get("intermediate_ctc_layers", []),
            stochastic_depth_rate=args.get("stochastic_depth_rate", 0.0),
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

    # Watch model with wandb
    wandb.watch(model, log="all", log_freq=100)

    n_classes = args["nClasses"] + 1  # +1 for CTC blank
    # All models append the blank token as the last class, so align CTCLoss to that index
    blank_idx = n_classes - 1

    # Label smoothing for better generalization
    label_smoothing = args.get("label_smoothing", 0.0)
    if label_smoothing > 0:
        # CTCLoss doesn't support label smoothing directly, so we'll apply it manually
        loss_ctc = torch.nn.CTCLoss(blank=blank_idx, reduction="none", zero_infinity=True)
    else:
        loss_ctc = torch.nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)

    if args.get("optimizer", "adam") == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=1e-8,
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

        # Unpack batch data (with or without diphone labels)
        batch_data = next(iter(trainLoader))
        if diphone_vocab is not None:
            X, y, X_len, y_len, dayIdx, y_diphone, y_diphone_len = batch_data
            y_diphone = y_diphone.to(device)
            y_diphone_len = y_diphone_len.to(device)
        else:
            X, y, X_len, y_len, dayIdx = batch_data
            y_diphone = None
            y_diphone_len = None

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

        # SpecAugment-style time masking (proven powerful for speech)
        time_mask_param = args.get("time_mask_param", 0)
        if time_mask_param > 0 and batch % 2 == 0:  # Apply to 50% of batches
            for i in range(X.shape[0]):
                # Random time mask
                t_length = X.shape[1]
                t_mask_len = min(time_mask_param, t_length // 4)  # Max 25% of sequence
                if t_mask_len > 0:
                    t_start = torch.randint(0, t_length - t_mask_len + 1, (1,)).item()
                    X[i, t_start:t_start + t_mask_len, :] = 0  # Mask with zeros

        # Compute prediction error
        model_type = args.get("model_type", "gru_baseline")

        if model_type == "advanced_multiscale":
            # Advanced multi-scale with auxiliary losses
            model_output = model(X, dayIdx, X_len, phoneme_labels=y, return_auxiliary_losses=(batch % 10 == 0))

            if len(model_output) == 3:
                log_probs, out_lens, aux_outputs = model_output
            else:
                log_probs, out_lens = model_output
                aux_outputs = {}

            # Main CTC loss
            loss = loss_ctc(log_probs, y, out_lens, y_len)

            if label_smoothing > 0:
                ctc_loss = torch.mean(loss)
                uniform_dist = torch.full_like(log_probs, -math.log(n_classes))
                kl_div = F.kl_div(log_probs, uniform_dist, reduction='batchmean', log_target=True)
                loss = (1 - label_smoothing) * ctc_loss + label_smoothing * kl_div
            else:
                loss = torch.sum(loss)

            # Add contrastive loss
            if 'contrastive_loss' in aux_outputs:
                contrastive_weight = args.get("contrastive_loss_weight", 0.1)
                loss = loss + contrastive_weight * aux_outputs['contrastive_loss']

        elif model_type == "multiscale_ctc":
            # Multi-scale CTC model
            # Optional: Contrastive pre-training phase
            use_contrastive = args.get("use_contrastive_pretraining", False)
            contrastive_warmup = args.get("contrastive_warmup_batches", 1000)

            if use_contrastive and batch < contrastive_warmup:
                # Contrastive pre-training phase
                contrastive_loss = model(
                    X, dayIdx, X_len,
                    return_contrastive_loss=True
                )
                loss = contrastive_loss
            else:
                # Regular CTC training (with optional diphone auxiliary head)
                model_output = model(X, dayIdx, X_len)

                # Check if model returned multi-scale outputs (Step 3)
                if len(model_output) == 7:
                    phone_log_probs, out_lens, diphone_log_probs, fast_phone_log_probs, fast_out_lens, slow_phone_log_probs, slow_out_lens = model_output
                    has_diphone = True
                    has_multiscale = True
                elif len(model_output) == 3:
                    phone_log_probs, out_lens, diphone_log_probs = model_output
                    has_diphone = True
                    has_multiscale = False
                else:
                    phone_log_probs, out_lens = model_output
                    has_diphone = False
                    has_multiscale = False

                # Check if we're using marginalization (proper DCoND)
                using_marginalization = model.use_marginalization if hasattr(model, 'use_marginalization') else False

                if using_marginalization:
                    # PROPER DCoND (Step 2): Joint CTC loss on BOTH phonemes and diphones
                    # phone_log_probs come from marginalized diphones
                    # Both losses backprop through diphone_head → encoder

                    # Phoneme CTC loss (on marginalized phoneme distribution)
                    phone_loss = loss_ctc(
                        phone_log_probs,
                        y,
                        out_lens,
                        y_len,
                    )

                    # Apply label smoothing if enabled
                    if label_smoothing > 0:
                        phone_ctc_loss = torch.mean(phone_loss)
                        uniform_dist = torch.full_like(phone_log_probs, -math.log(n_classes))
                        kl_div = F.kl_div(phone_log_probs, uniform_dist, reduction='batchmean', log_target=True)
                        phone_loss = (1 - label_smoothing) * phone_ctc_loss + label_smoothing * kl_div
                    else:
                        phone_loss = torch.sum(phone_loss)

                    # Diphone CTC loss (on primary diphone distribution)
                    if has_diphone and y_diphone is not None:
                        diphone_blank_idx = diphone_vocab.blank_id
                        if label_smoothing > 0:
                            loss_ctc_diphone = torch.nn.CTCLoss(blank=diphone_blank_idx, reduction="none", zero_infinity=True)
                        else:
                            loss_ctc_diphone = torch.nn.CTCLoss(blank=diphone_blank_idx, reduction="mean", zero_infinity=True)

                        diphone_loss = loss_ctc_diphone(
                            diphone_log_probs,
                            y_diphone,
                            out_lens,
                            y_diphone_len,
                        )

                        # Apply label smoothing to diphone loss if enabled
                        if label_smoothing > 0:
                            diphone_ctc_loss = torch.mean(diphone_loss)
                            uniform_dist = torch.full_like(diphone_log_probs, -math.log(diphone_vocab.get_vocab_size()))
                            kl_div = F.kl_div(diphone_log_probs, uniform_dist, reduction='batchmean', log_target=True)
                            diphone_loss = (1 - label_smoothing) * diphone_ctc_loss + label_smoothing * kl_div
                        else:
                            diphone_loss = torch.sum(diphone_loss)

                        # JOINT LOSS with alpha weighting (proper DCoND Step 2!)
                        # Both losses provide gradient signal to the same diphone_head
                            alpha = _get_diphone_alpha(
                                batch,
                                args["nBatch"],
                                schedule=args.get("diphone_alpha_schedule", "constant"),
                                constant_alpha=args.get("diphone_alpha_constant", 0.5),
                            )
                        loss = alpha * phone_loss + (1.0 - alpha) * diphone_loss

                        # STEP 3: Add multi-scale auxiliary losses if enabled
                        if has_multiscale:
                            # Auxiliary CTC loss on fast pathway (high temporal resolution)
                            fast_phone_loss = loss_ctc(
                                fast_phone_log_probs,
                                y,
                                fast_out_lens,
                                y_len,
                            )
                            if label_smoothing > 0:
                                fast_ctc_loss = torch.mean(fast_phone_loss)
                                uniform_dist = torch.full_like(fast_phone_log_probs, -math.log(n_classes))
                                kl_div = F.kl_div(fast_phone_log_probs, uniform_dist, reduction='batchmean', log_target=True)
                                fast_phone_loss = (1 - label_smoothing) * fast_ctc_loss + label_smoothing * kl_div
                            else:
                                fast_phone_loss = torch.sum(fast_phone_loss)

                            # Auxiliary CTC loss on slow pathway (low temporal resolution)
                            slow_phone_loss = loss_ctc(
                                slow_phone_log_probs,
                                y,
                                slow_out_lens,
                                y_len,
                            )
                            if label_smoothing > 0:
                                slow_ctc_loss = torch.mean(slow_phone_loss)
                                uniform_dist = torch.full_like(slow_phone_log_probs, -math.log(n_classes))
                                kl_div = F.kl_div(slow_phone_log_probs, uniform_dist, reduction='batchmean', log_target=True)
                                slow_phone_loss = (1 - label_smoothing) * slow_ctc_loss + label_smoothing * kl_div
                            else:
                                slow_phone_loss = torch.sum(slow_phone_loss)

                            # Lambda weights for auxiliary losses
                            lambda_fast = args.get("multiscale_lambda_fast", 0.3)
                            lambda_slow = args.get("multiscale_lambda_slow", 0.3)

                            # Total loss: main + auxiliary
                            loss = loss + lambda_fast * fast_phone_loss + lambda_slow * slow_phone_loss

                        # Log individual losses for monitoring
                        if batch % 100 == 0:
                            log_dict = {
                                "train/phone_loss": phone_loss.item(),
                                "train/diphone_loss": diphone_loss.item(),
                                "train/alpha": alpha,
                                "train/method": "marginalization_joint_multiscale" if has_multiscale else "marginalization_joint",
                            }
                            if has_multiscale:
                                log_dict.update({
                                    "train/fast_phone_loss": fast_phone_loss.item(),
                                    "train/slow_phone_loss": slow_phone_loss.item(),
                                    "train/lambda_fast": lambda_fast,
                                    "train/lambda_slow": lambda_slow,
                                })
                            wandb.log(log_dict)
                    else:
                        # Fallback: only phoneme loss if no diphone labels
                        loss = phone_loss
                else:
                    # OLD APPROACH: Separate phone and diphone heads
                    # Phoneme CTC loss
                    phone_loss = loss_ctc(
                        phone_log_probs,
                        y,
                        out_lens,
                        y_len,
                    )

                    # Apply label smoothing to phoneme loss if enabled
                    if label_smoothing > 0:
                        phone_ctc_loss = torch.mean(phone_loss)
                        uniform_dist = torch.full_like(phone_log_probs, -math.log(n_classes))
                        kl_div = F.kl_div(phone_log_probs, uniform_dist, reduction='batchmean', log_target=True)
                        phone_loss = (1 - label_smoothing) * phone_ctc_loss + label_smoothing * kl_div
                    else:
                        phone_loss = torch.sum(phone_loss)

                    # Diphone CTC loss (if enabled)
                    if has_diphone and y_diphone is not None:
                        # Create CTC loss for diphones (different blank token)
                        diphone_blank_idx = diphone_vocab.blank_id
                        if label_smoothing > 0:
                            loss_ctc_diphone = torch.nn.CTCLoss(blank=diphone_blank_idx, reduction="none", zero_infinity=True)
                        else:
                            loss_ctc_diphone = torch.nn.CTCLoss(blank=diphone_blank_idx, reduction="mean", zero_infinity=True)

                        diphone_loss = loss_ctc_diphone(
                            diphone_log_probs,
                            y_diphone,
                            out_lens,
                            y_diphone_len,
                        )

                        # Apply label smoothing to diphone loss if enabled
                        if label_smoothing > 0:
                            diphone_ctc_loss = torch.mean(diphone_loss)
                            uniform_dist = torch.full_like(diphone_log_probs, -math.log(diphone_vocab.get_vocab_size()))
                            kl_div = F.kl_div(diphone_log_probs, uniform_dist, reduction='batchmean', log_target=True)
                            diphone_loss = (1 - label_smoothing) * diphone_ctc_loss + label_smoothing * kl_div
                        else:
                            diphone_loss = torch.sum(diphone_loss)

                        # Joint loss with alpha weighting
                        alpha = _get_diphone_alpha(
                            batch,
                            args["nBatch"],
                            schedule=args.get("diphone_alpha_schedule", "constant"),
                            constant_alpha=args.get("diphone_alpha_constant", 0.5),
                        )
                        loss = alpha * phone_loss + (1.0 - alpha) * diphone_loss

                        # Log individual losses for monitoring
                        if batch % 100 == 0:
                            wandb.log({
                                "train/phone_loss": phone_loss.item(),
                                "train/diphone_loss": diphone_loss.item(),
                                "train/alpha": alpha,
                                "train/method": "separate_heads",
                            })
                    else:
                        # No diphone loss, use only phoneme loss
                        loss = phone_loss

        elif model_type == "transformer_ctc":
            model_output = model(X, dayIdx, X_len)
            # Handle both (log_probs, out_lens) and (log_probs, out_lens, intermediate_outputs)
            if len(model_output) == 3:
                log_probs, out_lens, intermediate_outputs = model_output
            else:
                log_probs, out_lens = model_output
                intermediate_outputs = {}

            loss = loss_ctc(
                log_probs,
                y,
                out_lens,
                y_len,
            )

            # Apply label smoothing if enabled
            if label_smoothing > 0:
                # CTC loss + uniform distribution over non-blank classes
                ctc_loss = torch.mean(loss)
                # Compute KL divergence from uniform distribution (label smoothing)
                uniform_dist = torch.full_like(log_probs, -math.log(n_classes))
                kl_div = F.kl_div(log_probs, uniform_dist, reduction='batchmean', log_target=True)
                loss = (1 - label_smoothing) * ctc_loss + label_smoothing * kl_div
            else:
                loss = torch.sum(loss)

            # Add intermediate CTC losses (improves gradient flow in deep models)
            if intermediate_outputs:
                intermediate_weight = args.get("intermediate_loss_weight", 0.3)
                for layer_name, intermediate_log_probs in intermediate_outputs.items():
                    intermediate_loss = loss_ctc(intermediate_log_probs, y, out_lens, y_len)
                    if label_smoothing > 0:
                        intermediate_loss = torch.mean(intermediate_loss)
                    else:
                        intermediate_loss = torch.sum(intermediate_loss)
                    loss = loss + intermediate_weight * intermediate_loss

        else:
            # GRU baseline
            pred = model.forward(X, dayIdx)
            out_lens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
            log_probs = pred.log_softmax(2).permute(1, 0, 2)

            loss = loss_ctc(
                log_probs,
                y,
                out_lens,
                y_len,
            )
            loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent explosion
        grad_clip_norm = args.get("grad_clip_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()
        scheduler.step()

        # Log training metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "train/batch": batch,
        }, step=batch)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for test_batch_data in testLoader:
                    # Handle both 5-tuple (no diphone) and 7-tuple (with diphone) formats
                    if diphone_vocab is not None:
                        X, y, X_len, y_len, testDayIdx, _, _ = test_batch_data
                    else:
                        X, y, X_len, y_len, testDayIdx = test_batch_data

                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    if model_type == "advanced_multiscale":
                        # Advanced multi-scale (same as multiscale_ctc for evaluation)
                        pred, adjustedLens = model(X, testDayIdx, X_len)
                        # pred is [T, B, C]

                        loss = loss_ctc(
                            pred,
                            y,
                            adjustedLens,
                            y_len,
                        )
                        loss = torch.sum(loss)
                        allLoss.append(loss.cpu().detach().numpy())

                        for iterIdx in range(pred.shape[1]):  # pred is [T, B, C]
                            decodedSeq = torch.argmax(
                                pred[0 : adjustedLens[iterIdx], iterIdx, :].clone().detach(),
                                dim=-1,
                            )  # [num_seq,]
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

                    elif model_type == "multiscale_ctc":
                        # Multi-scale CTC decoding (handle 2, 3, or 7 outputs)
                        model_output = model(X, testDayIdx, X_len)

                        # Unpack output - always use first two (phoneme predictions and lengths)
                        if len(model_output) == 7:
                            # Multi-scale with marginalization: 7 outputs
                            # (phone, lens, diphone, fast_phone, fast_lens, slow_phone, slow_lens)
                            pred, adjustedLens = model_output[0], model_output[1]
                        elif len(model_output) == 3:
                            # Marginalization without multi-scale: 3 outputs
                            pred, adjustedLens, _ = model_output  # Ignore diphone output during eval
                        else:
                            # Baseline: 2 outputs
                            pred, adjustedLens = model_output
                        # pred is [T, B, C]

                        loss = loss_ctc(
                            pred,
                            y,
                            adjustedLens,
                            y_len,
                        )
                        loss = torch.sum(loss)
                        allLoss.append(loss.cpu().detach().numpy())

                        for iterIdx in range(pred.shape[1]):  # pred is [T, B, C]
                            decodedSeq = torch.argmax(
                                pred[0 : adjustedLens[iterIdx], iterIdx, :].clone().detach(),
                                dim=-1,
                            )  # [num_seq,]
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

                    elif model_type == "transformer_ctc":
                        pred, adjustedLens = model(X, testDayIdx, X_len)
                        # pred already [T, B, C]

                        loss = loss_ctc(
                            pred,
                            y,
                            adjustedLens,
                            y_len,
                        )
                        loss = torch.sum(loss)
                        allLoss.append(loss.cpu().detach().numpy())

                        for iterIdx in range(pred.shape[1]):  # pred is [T, B, C]
                            decodedSeq = torch.argmax(
                                pred[0 : adjustedLens[iterIdx], iterIdx, :].clone().detach(),
                                dim=-1,
                            )  # [num_seq,]
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

                    else:
                        # GRU baseline
                        logits = model.forward(X, testDayIdx)
                        adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                            torch.int32
                        )
                        pred = logits.log_softmax(2).permute(1, 0, 2)

                        loss = loss_ctc(
                            pred,
                            y,
                            adjustedLens,
                            y_len,
                        )
                        loss = torch.sum(loss)
                        allLoss.append(loss.cpu().detach().numpy())

                        for iterIdx in range(pred.shape[1]):  # pred is [T, B, C]
                            decodedSeq = torch.argmax(
                                pred[0 : adjustedLens[iterIdx], iterIdx, :].clone().detach(),
                                dim=-1,
                            )  # [num_seq,]
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

                # Log evaluation metrics
                wandb.log({
                    "eval/loss": avgDayLoss,
                    "eval/cer": cer,
                    "eval/time_per_batch": time_per_batch,
                }, step=batch)

                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                wandb.log({"eval/best_cer": cer}, step=batch)
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)

    # Finish wandb run
    wandb.finish()


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    if args.get("model_type", "gru_baseline") == "transformer_ctc":
        model = NeuralTransformerCTCModel(
            n_channels=args["nInputFeatures"],
            n_classes=args["nClasses"] + 1,
            n_days=nInputLayers,
            use_wavelets=args.get("use_wavelets", True),
            n_scales=args.get("wavelet_n_scales", 4),
            wavelet_window_size=args.get("wavelet_window_size", 10),
            use_pac_features=args.get("use_pac_features", False),
            frontend_dim=args.get("frontend_dim", 512),
            latent_dim=args.get("latent_dim", 256),
            autoencoder_hidden_dim=args.get("autoencoder_hidden_dim", 128),
            transformer_layers=args.get("transformer_num_layers", 4),
            transformer_heads=args.get("transformer_n_heads", 4),
            transformer_ff_dim=args.get("transformer_dim_ff", 1024),
            transformer_dropout=args.get("transformer_dropout", 0.1),
            temporal_kernel=args.get("temporal_kernel", 0),
            temporal_stride=args.get("temporal_stride", 1),
            gaussian_smooth_width=args.get("gaussian_smooth_width", 0.0),
            use_conformer=args.get("use_conformer", False),
            conformer_conv_kernel=args.get("conformer_conv_kernel", 31),
            use_relative_pe=args.get("use_relative_pe", True),
            intermediate_ctc_layers=args.get("intermediate_ctc_layers", []),
            stochastic_depth_rate=args.get("stochastic_depth_rate", 0.0),
            device=device,
        ).to(device)
    else:
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
