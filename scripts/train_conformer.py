modelName = "conformer_improved"

args = {}
args["outputDir"] = "/data1/edward/Neural-Speech-Decoder/logs/speech_logs/" + modelName
args["datasetPath"] = "/data1/edward/Neural-Speech-Decoder/ptDecoder_ctc"
args["batchSize"] = 64
args["nBatch"] = 15000  # Increased from 10000 to allow Deep Head to converge
args["seed"] = 0

# Wandb logging
args["wandb_project"] = "neural-speech-decoder"
args["wandb_run_name"] = modelName
args["wandb_mode"] = "online"  # Can be 'online', 'offline', or 'disabled'

# Model type
args["model_type"] = "transformer_ctc"

# Data parameters (match GRU baseline)
args["nInputFeatures"] = 256
args["nClasses"] = 40

# Temporal processing (match GRU baseline)
args["temporal_kernel"] = 32
args["temporal_stride"] = 4
args["gaussian_smooth_width"] = 2.0

# Data augmentation (match GRU baseline)
args["whiteNoiseSD"] = 0.8
args["constantOffsetSD"] = 0.2

# Conformer architecture
args["frontend_dim"] = 1024
args["latent_dim"] = 1024
args["autoencoder_hidden_dim"] = 512
args["transformer_num_layers"] = 8
args["transformer_n_heads"] = 8
args["transformer_dim_ff"] = 2048
args["transformer_dropout"] = 0.3
args["conformer_conv_kernel"] = 31
args["drop_path_prob"] = 0.1  # Stochastic depth for better regularization

# Training optimization
args["optimizer"] = "adamw"
args["lrStart"] = 0.0004  # Reduced from 0.002 - safe zone to prevent loss explosion
args["lrEnd"] = 0.0001
args["weight_decay"] = 1e-3
args["warmup_steps"] = 1000  # Reduced from 4000 - shorter warmup for lower peak LR
args["label_smoothing"] = 0.1

# SpecAugment (time and feature masking)
args["use_spec_augment"] = True
args["spec_augment_freq_mask"] = 100  # Mask up to 100 features
args["spec_augment_time_mask"] = 40  # Mask up to 40 timesteps

# Intermediate CTC (InterCTC) for deep supervision
args["interctc_weight"] = 0.3  # Weight for intermediate loss

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
