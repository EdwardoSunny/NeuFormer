"""
Conformer v3 training script — maximum performance configuration.

Key changes from v2:
  - Rotary Position Embeddings (RoPE) replacing sinusoidal PE
  - 10 layers (was 8) for more model depth
  - 50k steps for thorough convergence
  - Higher peak LR with longer warmup
  - Stronger SpecAugment

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
args["nBatch"] = 50000  # More training for deeper model
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

# Data augmentation — slightly stronger
args["whiteNoiseSD"] = 1.0  # Increased from 0.8
args["constantOffsetSD"] = 0.3  # Increased from 0.2

# Conformer architecture (v3)
args["frontend_dim"] = 1024
args["latent_dim"] = 1024
args["autoencoder_hidden_dim"] = 512
args["transformer_num_layers"] = 10  # Deeper (was 8)
args["transformer_n_heads"] = 8
args["transformer_dim_ff"] = 4096  # 4x expansion
args["transformer_dropout"] = 0.2  # Lower dropout (more layers = more regularization)
args["conformer_conv_kernel"] = 31
args["drop_path_prob"] = 0.2  # Higher for deeper model

# Training optimization
args["optimizer"] = "adamw"
args["lrStart"] = 0.0006  # Slightly higher peak
args["lrEnd"] = 0.00003  # Low floor
args["weight_decay"] = 1e-3
args["warmup_steps"] = 3000  # Longer warmup for 50k steps
args["label_smoothing"] = 0.1

# SpecAugment — stronger
args["use_spec_augment"] = True
args["spec_augment_freq_mask"] = 120  # Wider masks
args["spec_augment_time_mask"] = 50  # Wider time masks

# Intermediate CTC
args["interctc_weight"] = 0.3

# v2 architecture features
args["autoencoder_residual"] = True

# v3 architecture features
args["use_rope"] = True  # Rotary Position Embeddings

# Training features
args["use_amp"] = True
args["use_ema"] = True
args["ema_decay"] = 0.9995  # Slower EMA for longer training

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
