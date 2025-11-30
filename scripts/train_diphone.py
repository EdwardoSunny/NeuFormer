
modelName = 'multiscale_diphone_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.0005  # Conservative LR (fixed from divergence)
args['lrEnd'] = 0.00005
args['nUnits'] = 1024
args['nBatch'] = 150000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

# ============================================================================
# MULTI-SCALE CTC + DIPHONE AUXILIARY HEAD
# ============================================================================
# Diphone head provides context-aware supervision to improve phoneme decoding
# Based on DCoND paper approach
# ============================================================================
args['model_type'] = 'multiscale_ctc'

# Multi-scale encoder architecture
args['latent_dim'] = 1024
args['transformer_num_layers'] = 6
args['transformer_n_heads'] = 8
args['transformer_dim_ff'] = 2048
args['transformer_dropout'] = 0.3
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# Training optimization (FIXED for stability)
args['optimizer'] = 'adamw'
args['weight_decay'] = 1e-3
args['warmup_steps'] = 10000  # Longer warmup for stability
args['grad_clip_norm'] = 0.5  # Stronger clipping
args['label_smoothing'] = 0.1

# Data augmentation
args['time_mask_param'] = 15  # Less aggressive (was 20)

# ===========================================================================
# DIPHONE WITH MARGINALIZATION (PROPER DCoND STEP 2)
# ===========================================================================
args['use_diphone_head'] = True  # Enable diphone head
args['diphone_vocab_path'] = '/home/edward/neural_seq_decoder/diphone_vocab.pkl'
args['use_diphone_marginalization'] = True  # Enable proper marginalization!

# DCoND Step 2 - Joint Loss Architecture:
#   - Model has ONLY diphone_output head (no phone_output)
#   - Phoneme probs: P(phoneme_j) = sum_i P(diphone_ij) via marginalization
#   - JOINT CTC LOSS: loss = α * phone_loss + (1-α) * diphone_loss
#     * phone_loss: CTC on marginalized phoneme distribution
#     * diphone_loss: CTC on primary diphone distribution
#     * Both losses backprop through same diphone_head → stronger gradient signal!
#   - Decoding: Use marginalized phoneme probs (same as before)

# Alpha scheduling (controls phone vs diphone loss weighting):
args['diphone_alpha_schedule'] = 'constant'  # Options: 'constant' or 'scheduled'
# - 'constant': α = 0.5 throughout (50/50 weighting)
# - 'scheduled': Start with more diphone focus, end with more phone focus
#   * First 20%: α = 0.3 (lean on diphones for context)
#   * Middle 60%: α ramps 0.3 → 0.7
#   * Last 20%: α = 0.8 (focus on phonemes)

# Benefits of joint loss over single phoneme loss:
#   1. Diphone CTC provides direct supervision on diphone predictions
#   2. Phoneme CTC ensures marginalized predictions match targets

# ===========================================================================
# MULTI-SCALE CTC HEADS (DCoND STEP 3)
# ===========================================================================
args['use_multiscale_ctc'] = True  # Enable Step 3: auxiliary CTC heads on fast/slow pathways

# Lambda weights for auxiliary losses:
#   total_loss = main_loss + λ_fast * fast_loss + λ_slow * slow_loss
args['multiscale_lambda_fast'] = 0.3  # Weight for fast pathway (stride 2, ~75 timesteps)
args['multiscale_lambda_slow'] = 0.3  # Weight for slow pathway (stride 8, ~19 timesteps)

# Benefits of multi-scale CTC (Step 3):
#   1. Direct supervision at each temporal scale (fast, medium, slow)
#   2. Stronger gradient flow to all encoder layers
#   3. Fast pathway learns fine-grained temporal patterns
#   4. Slow pathway learns long-range context
#   5. Improved training stability and faster convergence
#   6. Expected to improve CER by additional 10-20% beyond Step 2

# Without marginalization (OLD baseline, for comparison):
#   - Set use_diphone_marginalization = False
#   - Model has separate phone_output and diphone_output heads
#   - Two independent predictions (diphone doesn't help phoneme)

# Note: Model has ~241M parameters (no separate phoneme head needed!)

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
