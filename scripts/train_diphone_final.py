"""
FINAL OPTIMIZED Training Script (balanced for stability and speed)

Key choices:
1. Constant alpha with phoneme bias (0.6) - speeds CER without over-shooting
2. Multi-scale lambda 0.1 - auxiliary heads help without dominating
3. Light label smoothing 0.01 - small stability boost
4. Moderate warmup 7k - reach effective LR with gentle ramp
5. Grad clip 1.0 - stable updates
"""

modelName = 'multiscale_diphone_final_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64

# Requested LR peak
args['lrStart'] = 0.0015
args['lrEnd'] = 0.00005
args['nUnits'] = 1024
args['nBatch'] = 200000
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

# Model architecture
args['model_type'] = 'multiscale_ctc'
args['latent_dim'] = 1024
args['transformer_num_layers'] = 6
args['medium_encoder_layers'] = 8  # Increased capacity for main pathway
args['aux_encoder_layers'] = 4     # Deeper auxiliary pathways
args['transformer_n_heads'] = 8
args['transformer_dim_ff'] = 2048
args['transformer_dropout'] = 0.1
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# Training optimization
args['optimizer'] = 'adamw'
args['weight_decay'] = 1e-3

# ===========================================================================
# FIX #4: Warmup per request (10k) - smoother ramp
# ===========================================================================
args['warmup_steps'] = 10000

# ===========================================================================
# FIX #5: More conservative grad clipping (1.0 not 1.5)
# ===========================================================================
args['grad_clip_norm'] = 1.0  # More stable

# ===========================================================================
# FIX #3: Light label smoothing for stability
# ===========================================================================
args['label_smoothing'] = 0.01

args['time_mask_param'] = 20

# Diphone configuration
args['use_diphone_head'] = True
args['diphone_vocab_path'] = '/home/edward/neural_seq_decoder/diphone_vocab.pkl'
args['use_diphone_marginalization'] = True

# ===========================================================================
# FIX #1: CONSTANT alpha (phoneme-biased) - prevents loss instability
# ===========================================================================
args['diphone_alpha_schedule'] = 'constant'  # Stable weighting throughout
args['diphone_alpha_constant'] = 0.65  # Bias toward phoneme loss early

# ===========================================================================
# FIX #2: Balanced auxiliary weights (0.1 keeps focus on main head)
# ===========================================================================
args['use_multiscale_ctc'] = True
args['multiscale_lambda_fast'] = 0.1  # Auxiliary heads still help but less dominant
args['multiscale_lambda_slow'] = 0.1  # Auxiliary heads still help but less dominant

print("=" * 80)
print("FINAL OPTIMIZED CONFIGURATION")
print("=" * 80)
print(f"Model: {modelName}")
print()
print("ALL LOGIC FIXES APPLIED:")
print(f"  1. ✓ diphone_alpha = 'constant' (phoneme-biased 0.65 - STABLE)")
print(f"  2. ✓ multiscale_lambda = 0.1 (auxiliary but not dominant)")
print(f"  3. ✓ label_smoothing = 0.01 (light smoothing for stability)")
print(f"  4. ✓ warmup_steps = 10k (reach peak LR per request)")
print(f"  5. ✓ grad_clip_norm = 1.0 (not 1.5 - MORE STABLE)")
print()
print("WHY THESE FIXES:")
print("  Issue 1: Scheduled alpha caused loss to increase (moving target)")
print("  Issue 2: Aux weight too weak/strong; 0.1 keeps focus on main head")
print("  Issue 3: Tiny smoothing helps stability without slowing too much")
print("  Issue 4: Very long warmup slows learning; moderate ramp balances speed/stability")
print("  Issue 5: 1.5 grad clip less stable than 1.0")
print()
print("EXPECTED: Faster early CER drop, stable training, better final CER")
print("TARGET: 0.15-0.22 CER (realistic with all fixes)")
print("=" * 80)
print()

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
