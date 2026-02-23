"""
Conformer v4a — v3 baseline + depthwise frontend ONLY.

Isolating the depthwise frontend to see if it helps or hurts.
v4 (all 3 changes at once) was at CER 0.218 at step 4700, while
v2/v3 were at ~0.18 at the same point. Need to find what's hurting.

This is exactly v3 (which reached 0.1387 best CER) with one change:
  - use_depthwise_frontend=True (B2T-style electrode-independent first conv)

No decoder layers, no char CTC, no other changes.
"""

modelName = "conformer_v4a"

args = {}
args["outputDir"] = "/data1/edward/Neural-Speech-Decoder/logs/speech_logs/" + modelName
args["datasetPath"] = "/data1/edward/Neural-Speech-Decoder/ptDecoder_ctc"
args["batchSize"] = 64
args["nBatch"] = 30000  # Same as v2/v3
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

# Data augmentation — same as v3
args["whiteNoiseSD"] = 0.8
args["constantOffsetSD"] = 0.2
args["mult_noise_sd"] = 0.1
args["temporal_shift_max"] = 7

# Conformer architecture — identical to v2/v3
args["frontend_dim"] = 1024
args["latent_dim"] = 1024
args["autoencoder_hidden_dim"] = 512
args["transformer_num_layers"] = 8
args["transformer_n_heads"] = 8
args["transformer_dim_ff"] = 4096
args["transformer_dropout"] = 0.25
args["conformer_conv_kernel"] = 31
args["drop_path_prob"] = 0.15

# === ONLY NEW THING: B2T-style depthwise frontend ===
args["use_depthwise_frontend"] = True
args["depthwise_hidden_dim"] = 1024

# No decoder layers (isolating frontend)
# No char CTC (isolating frontend)

# Training optimization — identical to v2/v3
args["optimizer"] = "adamw"
args["lrStart"] = 0.0005
args["lrEnd"] = 0.00005
args["weight_decay"] = 1e-3
args["warmup_steps"] = 2000
args["label_smoothing"] = 0.1

# SpecAugment — same as v3
args["use_spec_augment"] = True
args["spec_augment_freq_mask"] = 110
args["spec_augment_time_mask"] = 45

# Intermediate CTC
args["interctc_weight"] = 0.3

# v2 architecture features
args["autoencoder_residual"] = True

# No RoPE
args["use_rope"] = False

# Training features — same as v2/v3
args["use_amp"] = True
args["use_ema"] = True
args["ema_decay"] = 0.999

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
