# Training Configuration: Conformer Speech BCI Decoder

## Model Architecture Parameters

```python
# Model Type
model_type: 'transformer_ctc'

# Architecture Dimensions
frontend_dim: 1024              # Output dimension of neural frontend
latent_dim: 1024                # Latent dimension throughout Conformer blocks
autoencoder_hidden_dim: 512     # Hidden dimension in autoencoder bottleneck

# Conformer Configuration
transformer_num_layers: 8       # Number of Conformer blocks
transformer_n_heads: 8          # Attention heads per block (128 dim each)
transformer_dim_ff: 2048        # Feed-forward intermediate dimension
transformer_dropout: 0.3        # Dropout probability throughout model
conformer_conv_kernel: 31       # Depthwise convolution kernel size
drop_path_prob: 0.1            # Stochastic depth (DropPath) probability

# Input/Output
nInputFeatures: 256             # Number of neural recording channels
nClasses: 40                    # Number of phonemes (+ 1 blank for CTC)
```

## Temporal Processing (Frontend)

```python
# Preprocessing (matching GRU baseline)
temporal_kernel: 32             # Strided convolution kernel size
temporal_stride: 4              # Temporal downsampling factor (4×)
gaussian_smooth_width: 2.0      # Gaussian smoothing sigma

# Effect: Input length T → ~(T - 32) / 4
# Example: 1000 timesteps → ~242 timesteps (4× compression)
```

## Data Augmentation

```python
# Noise Augmentation (on GPU, applied to neural features)
whiteNoiseSD: 0.8              # White noise standard deviation
constantOffsetSD: 0.2          # Constant offset per trial (simulates drift)

# SpecAugment (applied to latent features, training only)
use_spec_augment: True
spec_augment_freq_mask: 100    # Max feature dimensions to mask
spec_augment_time_mask: 40     # Max timesteps to mask
# 2 frequency masks + 2 time masks applied per sample
```

## Optimization

```python
# Optimizer
optimizer: 'adamw'
lrStart: 0.0004                # Peak learning rate (reduced from 0.002)
lrEnd: 0.0001                  # Final learning rate (not used with cosine)
weight_decay: 1e-3             # L2 regularization weight
betas: (0.9, 0.999)            # Adam beta parameters
eps: 1e-6                      # Epsilon (increased for mixed precision stability)

# Learning Rate Schedule
warmup_steps: 1000             # Linear warmup steps (reduced from 4000)
# After warmup: Cosine annealing decay over remaining steps
# Formula: lr = peak_lr × 0.5 × (1 + cos(π × progress))

# Training Duration
nBatch: 15000                  # Total training batches (increased from 10000)
batchSize: 64                  # Batch size
```

## Loss Function & Regularization

```python
# CTC Loss Configuration
blank_idx: 0                   # Blank token index for CTC
reduction: 'none'              # Per-sample loss (for label smoothing)
zero_infinity: True            # Stability for invalid paths

# Label Smoothing
label_smoothing: 0.1           # Mix CTC loss with uniform distribution
# Loss = (1 - 0.1) × CTC_loss + 0.1 × KL_div(pred || uniform)

# Intermediate CTC (Deep Supervision)
interctc_weight: 0.3           # Weight for intermediate loss at layer 4
# Total loss = 0.7 × main_CTC + 0.3 × inter_CTC (layer 4)

# Gradient Clipping
max_grad_norm: 1.0             # Maximum gradient norm (prevents explosion)
```

## Regularization Strategy Summary

| Technique | Configuration | Purpose |
|-----------|---------------|---------|
| **Dropout** | 0.3 | General regularization throughout model |
| **Stochastic Depth** | 0.1 | Randomly drop residual paths during training |
| **Label Smoothing** | 0.1 | Prevent overconfidence, improve generalization |
| **Weight Decay** | 1e-3 | L2 regularization on all parameters |
| **SpecAugment** | freq=100, time=40 | Force reliance on context, not local features |
| **InterCTC** | weight=0.3 | Deep supervision for middle layers |
| **White Noise** | σ=0.8 | Robustness to neural recording noise |
| **Constant Offset** | σ=0.2 | Robustness to recording drift |
| **Gradient Clipping** | norm=1.0 | Training stability |

## Training Schedule

```
Total iterations: 15,000 batches
Evaluation: Every 100 batches
Batch size: 64 samples
Estimated total samples seen: 960,000

Learning rate schedule:
├─ Steps 0-1000:     Linear warmup (0 → 0.0004)
└─ Steps 1000-15000: Cosine decay (0.0004 → ~0)

Time per batch: ~7.3 seconds (on single GPU)
Total training time: ~30 hours
```

## Evaluation Metrics

```python
# Primary Metric
CER (Character Error Rate): Edit distance / sequence length
# Best achieved: 0.16 (16% error rate)

# Secondary Metrics
- CTC Loss (validation set)
- Per-batch training time
- Gradient norm (monitoring for stability)
```

## Key Design Decisions

### 1. **Conservative Learning Rate (0.0004)**
- Reduced from initial 0.002 to prevent loss explosion
- Critical for Conformer stability with deep architecture
- Shorter warmup (1000 steps) compensates for lower peak

### 2. **Extended Training (15,000 batches)**
- Increased from 10,000 to allow deep classification head to converge
- Non-linear head requires more iterations than simple linear projection
- Cosine schedule ensures gradual refinement in later stages

### 3. **Heavy Regularization**
- Multiple complementary techniques prevent overfitting
- SpecAugment + DropPath + Label Smoothing work synergistically
- Essential for ~45M parameter model with limited BCI data

### 4. **InterCTC Deep Supervision**
- Intermediate loss at layer 4 (middle of 8 layers)
- Provides gradient signal directly to early/middle layers
- Improves training stability and convergence speed
- 30% weight balances main and intermediate objectives

### 5. **Matched Preprocessing to GRU Baseline**
- Identical Gaussian smoothing, temporal kernel/stride
- Ensures fair comparison between architectures
- Preserves domain knowledge from prior work

## Hardware & Implementation

```python
device: 'cuda'                 # Single GPU training
precision: 'fp32'              # Full precision (fp16 compatible)
num_workers: 0                 # DataLoader workers (on GPU augmentation)
pin_memory: True               # Faster host-to-device transfer

# Logging
wandb_project: 'neural-speech-decoder'
wandb_mode: 'online'           # Real-time experiment tracking
log_frequency: 1               # Log training loss every batch
eval_frequency: 100            # Evaluate every 100 batches
```

## Reproducibility

```python
seed: 0                        # Random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
# Note: Full determinism requires additional CUDA settings
```

---

## Summary

The Conformer model is trained with a **conservative but effective** strategy: moderate learning rate (0.0004), aggressive regularization (dropout, DropPath, label smoothing, SpecAugment), deep supervision (InterCTC), and extended training (15,000 batches). The configuration prioritizes **stability and generalization** over rapid convergence, crucial for the limited data regime typical of BCI applications. The 4× temporal downsampling balances computational efficiency with temporal resolution, while matched preprocessing ensures fair comparison to the GRU baseline. This configuration achieves **16% CER**, demonstrating that careful hyperparameter tuning enables transformer architectures to excel at neural decoding tasks.
