"""
Conformer v4 training script — B2T-inspired architecture on v2 recipe.

Strategy: keep v2's proven training recipe and only add the B2T-inspired
architectural changes that address the acoustic model quality gap:

1. **Depthwise-first frontend** (B2T-style):
   - Depthwise Conv(k=7, s=2, groups=256) → Highway(2) → Conv(k=3, s=2) → Highway(2)
   - Processes each electrode independently before mixing channels
   - Neurophysiologically motivated: different electrodes record different brain areas

2. **Deep decoder head** (2 Conformer layers between encoder and CTC):
   - B2T uses 4-layer bidirectional Mamba (~38M params). We use 2 Conformer layers.
   - Adds output-specific capacity without changing the encoder.

3. **Dual-task CTC** (phoneme + character):
   - Shared encoder, two output heads
   - Forces richer intermediate representations

4. **v3 augmentations retained**: multiplicative noise, temporal shift

Training recipe is identical to v2:
  - 8 Conformer layers, no RoPE
  - LR 5e-4 → 5e-5, cosine schedule, 2k warmup
  - AdamW with weight_decay=1e-3
  - 30k steps, batch 64
  - EMA 0.999, AMP
"""

modelName = "conformer_v4"

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

# Data augmentation — v2 baseline + B2T augmentations (same as v3)
args["whiteNoiseSD"] = 0.8
args["constantOffsetSD"] = 0.2
args["mult_noise_sd"] = 0.1  # B2T-inspired multiplicative noise
args["temporal_shift_max"] = 7  # B2T-inspired temporal shift

# === Conformer architecture — v2 base + B2T-inspired additions ===

# Encoder (identical to v2)
args["frontend_dim"] = 1024
args["latent_dim"] = 1024
args["autoencoder_hidden_dim"] = 512
args["transformer_num_layers"] = 8  # Same as v2 (10 was worse)
args["transformer_n_heads"] = 8
args["transformer_dim_ff"] = 4096  # 4x expansion, same as v2
args["transformer_dropout"] = 0.25  # Same as v2
args["conformer_conv_kernel"] = 31
args["drop_path_prob"] = 0.15  # Same as v2

# NEW: B2T-style depthwise-first frontend
args["use_depthwise_frontend"] = True
args["depthwise_hidden_dim"] = 1024  # 1024/256 = 4 channels per electrode

# NEW: Deep decoder head (2 Conformer layers before CTC projection)
args["decoder_layers"] = 2
args["decoder_ff_dim"] = 2048  # Smaller than encoder FF for efficiency

# Training optimization — identical to v2
args["optimizer"] = "adamw"
args["lrStart"] = 0.0005
args["lrEnd"] = 0.00005
args["weight_decay"] = 1e-3
args["warmup_steps"] = 2000
args["label_smoothing"] = 0.1

# SpecAugment — slightly stronger than v2 (same as v3)
args["use_spec_augment"] = True
args["spec_augment_freq_mask"] = 110
args["spec_augment_time_mask"] = 45

# Intermediate CTC
args["interctc_weight"] = 0.3

# v2 architecture features
args["autoencoder_residual"] = True

# No RoPE (hurt performance on this dataset)
args["use_rope"] = False

# NEW: Dual-task CTC (character-level head)
args["use_char_ctc"] = True
args["char_ctc_weight"] = 0.3  # 30% char loss + 70% phoneme loss

# Training features — same as v2
args["use_amp"] = True
args["use_ema"] = True
args["ema_decay"] = 0.999

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
