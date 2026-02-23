"""
Conformer v3 training script — RoPE + deeper model.

Key changes from v2:
  - Rotary Position Embeddings (RoPE) replacing sinusoidal PE
  - 10 layers (was 8) for more model depth
  - 30k steps (same as v2 — 50k was overfitting)
  - Same LR/dropout as v2 (conservative — deeper model already adds capacity)
  - Slightly stronger SpecAugment + noise

All v2 improvements are retained:
  - 4x FF expansion, linearly increasing DropPath
  - DropPath on ConvModule, AutoEncoder residual+LN
  - Proper data iteration, cosine to lrEnd
  - AMP, EMA
"""

modelName = "conformer_v3"

args = {}
args["outputDir"] = "/data1/edward/Neural-Speech-Decoder/logs/speech_logs/" + modelName
args["datasetPath"] = "/data1/edward/Neural-Speech-Decoder/ptDecoder_ctc"
args["batchSize"] = 64
args["nBatch"] = 30000  # Same as v2 — 50k was overfitting
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

# Data augmentation — slightly stronger than v2
args["whiteNoiseSD"] = 0.9  # Slight increase from 0.8
args["constantOffsetSD"] = 0.25  # Slight increase from 0.2

# Conformer architecture (v3)
args["frontend_dim"] = 1024
args["latent_dim"] = 1024
args["autoencoder_hidden_dim"] = 512
args["transformer_num_layers"] = 10  # Deeper (was 8)
args["transformer_n_heads"] = 8
args["transformer_dim_ff"] = 4096  # 4x expansion
args["transformer_dropout"] = 0.25  # Same as v2 (0.2 was too low, overfitting)
args["conformer_conv_kernel"] = 31
args["drop_path_prob"] = 0.2  # Higher for deeper model

# Training optimization
args["optimizer"] = "adamw"
args["lrStart"] = 0.0005  # Same as v2 (0.0006 was too aggressive)
args["lrEnd"] = 0.00005  # Same as v2
args["weight_decay"] = 1e-3
args["warmup_steps"] = 2000  # Same as v2
args["label_smoothing"] = 0.1

# SpecAugment — slightly stronger than v2
args["use_spec_augment"] = True
args["spec_augment_freq_mask"] = 110  # Slightly wider than v2's 100
args["spec_augment_time_mask"] = 45  # Slightly wider than v2's 40

# Intermediate CTC
args["interctc_weight"] = 0.3

# v2 architecture features
args["autoencoder_residual"] = True

# v3 architecture features
args["use_rope"] = True  # Rotary Position Embeddings

# Training features
args["use_amp"] = True
args["use_ema"] = True
args["ema_decay"] = 0.999  # Same as v2

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
