"""
Conformer v2 training script — improved architecture and training.

Key changes from v1:
  - FF expansion 2x → 4x (2048 → 4096) for more model capacity
  - Linearly increasing DropPath schedule (handled by model now)
  - DropPath applied to ConformerConvModule (handled by model now)
  - AutoEncoder has residual + LayerNorm + dropout (handled by model now)
  - Proper data iteration (no more next(iter()) bug)
  - Cosine LR decays to lrEnd (not zero)
  - AMP (mixed precision) for ~2x training speed
  - EMA of model weights for better generalization
  - 30k steps (2x more training with proper data cycling)
  - Higher DropPath (0.15) since model is larger
  - Slightly reduced dropout (0.25) since we have more regularization now
"""

modelName = "conformer_v2"

args = {}
args["outputDir"] = "/data1/edward/Neural-Speech-Decoder/logs/speech_logs/" + modelName
args["datasetPath"] = "/data1/edward/Neural-Speech-Decoder/ptDecoder_ctc"
args["batchSize"] = 64
args["nBatch"] = 30000  # 2x more training with proper epoch cycling
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

# Data augmentation
args["whiteNoiseSD"] = 0.8
args["constantOffsetSD"] = 0.2

# Conformer architecture (v2 improvements)
args["frontend_dim"] = 1024
args["latent_dim"] = 1024
args["autoencoder_hidden_dim"] = 512
args["transformer_num_layers"] = 8
args["transformer_n_heads"] = 8
args["transformer_dim_ff"] = 4096  # 4x expansion (was 2x = 2048)
args["transformer_dropout"] = (
    0.25  # Slightly reduced (more regularization from DropPath/EMA)
)
args["conformer_conv_kernel"] = 31
args["drop_path_prob"] = (
    0.15  # Slightly higher for larger model (linearly increasing schedule now)
)

# Training optimization
args["optimizer"] = "adamw"
args["lrStart"] = 0.0005  # Slightly higher — more steps to converge
args["lrEnd"] = 0.00005  # Cosine decays to this (not zero)
args["weight_decay"] = 1e-3
args["warmup_steps"] = 2000  # Longer warmup for 30k total steps
args["label_smoothing"] = 0.1

# SpecAugment
args["use_spec_augment"] = True
args["spec_augment_freq_mask"] = 100
args["spec_augment_time_mask"] = 40

# Intermediate CTC
args["interctc_weight"] = 0.3

# v2 architecture features
args["autoencoder_residual"] = True  # Residual + LayerNorm in AutoEncoder

# New v2 training features
args["use_amp"] = True  # Mixed precision for ~2x speedup
args["use_ema"] = True  # EMA of model weights
args["ema_decay"] = 0.999

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
