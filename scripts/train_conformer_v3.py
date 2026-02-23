"""
Conformer v3 training script — v2 architecture + B2T augmentations.

v2 at step 5100 had 0.18 CER. Previous v3 attempts with RoPE + 10 layers
plateaued at 0.24 CER at step 5000 — deeper model + RoPE hurt on this dataset.

Strategy: keep v2's proven architecture (8 layers, no RoPE) and only add
the B2T-inspired augmentations that are low-risk:
  - Multiplicative noise (simulates electrode gain drift)
  - Random temporal shift (alignment robustness)
  - Slightly stronger SpecAugment

Everything else identical to v2.
"""

modelName = "conformer_v3"

args = {}
args["outputDir"] = "/data1/edward/Neural-Speech-Decoder/logs/speech_logs/" + modelName
args["datasetPath"] = "/data1/edward/Neural-Speech-Decoder/ptDecoder_ctc"
args["batchSize"] = 64
args["nBatch"] = 30000  # Same as v2
args["seed"] = 42

# Wandb logging
args["wandb_project"] = "neural-speech-decoder"
args["wandb_run_name"] = modelName
args["wandb_mode"] = "online"

# Model type
args["model_type"] = "transformer_ctc"

# Data parameters
args["nInputFeatures"] = 256
args["nClasses"] = 40

# Temporal processing
args["temporal_kernel"] = 32
args["temporal_stride"] = 4
args["gaussian_smooth_width"] = 2.0

# Data augmentation — v2 baseline + B2T augmentations
args["whiteNoiseSD"] = 0.8  # Same as v2
args["constantOffsetSD"] = 0.2  # Same as v2
args["mult_noise_sd"] = 0.1  # NEW: multiplicative noise from B2T
args["temporal_shift_max"] = 7  # NEW: random temporal shift from B2T

# Conformer architecture — identical to v2
args["frontend_dim"] = 1024
args["latent_dim"] = 1024
args["autoencoder_hidden_dim"] = 512
args["transformer_num_layers"] = 8  # Same as v2 (10 was worse)
args["transformer_n_heads"] = 8
args["transformer_dim_ff"] = 4096  # 4x expansion, same as v2
args["transformer_dropout"] = 0.25  # Same as v2
args["conformer_conv_kernel"] = 31
args["drop_path_prob"] = 0.15  # Same as v2

# Training optimization — identical to v2
args["optimizer"] = "adamw"
args["lrStart"] = 0.0005
args["lrEnd"] = 0.00005
args["weight_decay"] = 1e-3
args["warmup_steps"] = 2000
args["label_smoothing"] = 0.1

# SpecAugment — slightly stronger than v2
args["use_spec_augment"] = True
args["spec_augment_freq_mask"] = 110  # v2 was 100
args["spec_augment_time_mask"] = 45  # v2 was 40

# Intermediate CTC
args["interctc_weight"] = 0.3

# v2 architecture features
args["autoencoder_residual"] = True

# No RoPE (hurt performance on this dataset)
args["use_rope"] = False

# Training features — same as v2
args["use_amp"] = True
args["use_ema"] = True
args["ema_decay"] = 0.999

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
