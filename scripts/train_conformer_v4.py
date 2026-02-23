"""
Conformer v4 training script — targeting <18% WER.

Key changes from v3:
  - Dual-task CTC: character-level CTC head alongside phoneme CTC
    (shared encoder, forces richer representations)
  - Multiplicative noise augmentation (simulates electrode gain changes)
  - Random temporal shift (robustness to alignment jitter)
  - beta2=0.95 (faster optimizer adaptation, used by LLaMA/GPT)
  - weight_decay=0.1 (stronger regularization, matches B2T_Model)
  - 30% warmup ratio (more stable early training)
  - 12 layers (deeper model, more capacity)
  - 60k steps for thorough convergence

All v2/v3 improvements retained:
  - 4x FF expansion, linearly increasing DropPath
  - DropPath on ConvModule, AutoEncoder residual+LN
  - RoPE, AMP, EMA
  - Intermediate CTC, label smoothing
"""

modelName = "conformer_v4"

args = {}
args["outputDir"] = "/data1/edward/Neural-Speech-Decoder/logs/speech_logs/" + modelName
args["datasetPath"] = "/data1/edward/Neural-Speech-Decoder/ptDecoder_ctc"
args["batchSize"] = 64
args["nBatch"] = 60000  # More training for deeper model + dual task
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

# --- v4 Data augmentation ---
args["whiteNoiseSD"] = 1.0
args["constantOffsetSD"] = 0.3
args["mult_noise_sd"] = 0.1  # Multiplicative noise (NEW)
args["temporal_shift_max"] = 8  # Random temporal shift 0-8 steps (NEW)

# Conformer architecture (v4)
args["frontend_dim"] = 1024
args["latent_dim"] = 1024
args["autoencoder_hidden_dim"] = 512
args["transformer_num_layers"] = 12  # Deeper (was 10 in v3)
args["transformer_n_heads"] = 8
args["transformer_dim_ff"] = 4096  # 4x expansion
args["transformer_dropout"] = (
    0.15  # Lower (deeper model = more implicit regularization)
)
args["conformer_conv_kernel"] = 31
args["drop_path_prob"] = 0.2  # Higher for deeper model

# --- v4 Optimizer (matches modern LLM practice) ---
args["optimizer"] = "adamw"
args["lrStart"] = 0.0005
args["lrEnd"] = 0.00002
args["beta2"] = 0.95  # Faster adaptation (NEW, was 0.999)
args["weight_decay"] = 0.1  # Stronger regularization (NEW, was 1e-3)
args["warmup_steps"] = 18000  # 30% of 60k steps (NEW)
args["label_smoothing"] = 0.1

# SpecAugment — strong
args["use_spec_augment"] = True
args["spec_augment_freq_mask"] = 120
args["spec_augment_time_mask"] = 50

# Intermediate CTC
args["interctc_weight"] = 0.3

# v2/v3 architecture features
args["autoencoder_residual"] = True
args["use_rope"] = True

# --- v4 Dual-task CTC ---
args["use_char_ctc"] = True  # Character-level CTC head (NEW)
args["char_ctc_weight"] = 0.3  # 30% char loss + 70% phoneme loss (NEW)

# Training features
args["use_amp"] = True
args["use_ema"] = True
args["ema_decay"] = 0.9995

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
