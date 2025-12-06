# Conformer-based Neural Decoder Architecture Diagram

```
INPUT: Neural Recordings
[Batch, Time, 256 channels]
         |
         v
┌─────────────────────────────────────────────────────────────┐
│  DAY-SPECIFIC LINEAR TRANSFORM                              │
│  • Per-day weight matrix [n_days, 256, 256]                 │
│  • Per-day bias [n_days, 1, 256]                            │
│  • Initialized as identity                                  │
│  Purpose: Handle cross-session neural drift                 │
└─────────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────────┐
│  NEURAL FRONTEND                                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 1. Gaussian Smoothing (σ=2.0, kernel=9)              │  │
│  │    • Reduces high-frequency noise                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                    ↓                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 2. Strided Temporal Conv (k=32, stride=4)            │  │
│  │    • Depthwise (channel-wise)                         │  │
│  │    • Downsamples: T → T/4                             │  │
│  └───────────────────────────────────────────────────────┘  │
│                    ↓                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 3. Linear Projection: 256 → 1024                     │  │
│  │    • LayerNorm + Dropout(0.3)                         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         |
         v [Batch, T/4, 1024]
┌─────────────────────────────────────────────────────────────┐
│  AUTOENCODER BOTTLENECK (MiSTR-style)                       │
│  Linear(1024 → 512) → ReLU → Linear(512 → 1024)            │
│  Purpose: Information compression & regularization          │
└─────────────────────────────────────────────────────────────┘
         |
         v [Batch, T/4, 1024]
┌─────────────────────────────────────────────────────────────┐
│  SPECAUGMENT (Training only)                                │
│  • Frequency masking: 2 masks, width ≤ 100 features         │
│  • Time masking: 2 masks, width ≤ 40 timesteps              │
│  Purpose: Force reliance on global context                  │
└─────────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────────┐
│  POSITIONAL ENCODING                                        │
│  Sinusoidal encoding (max_len=5000)                         │
│  PE(pos,2i) = sin(pos/10000^(2i/d))                         │
│  PE(pos,2i+1) = cos(pos/10000^(2i/d))                       │
└─────────────────────────────────────────────────────────────┘
         |
         v [Batch, T/4, 1024]
╔═════════════════════════════════════════════════════════════╗
║  CONFORMER BLOCKS (×8 layers)                               ║
║                                                             ║
║  ┌─────────────────────────────────────────────────────┐   ║
║  │ CONFORMER BLOCK (repeated 8 times)                  │   ║
║  │                                                      │   ║
║  │  Input [B, T/4, 1024]                               │   ║
║  │    ↓                                                 │   ║
║  │  ┌────────────────────────────────────────────────┐ │   ║
║  │  │ 1. Feed-Forward Module (half-step)             │ │   ║
║  │  │    LayerNorm                                    │ │   ║
║  │  │    Linear(1024 → 2048)                          │ │   ║
║  │  │    SiLU activation                              │ │   ║
║  │  │    Dropout(0.3)                                 │ │   ║
║  │  │    Linear(2048 → 1024)                          │ │   ║
║  │  │    Dropout(0.3)                                 │ │   ║
║  │  └────────────────────────────────────────────────┘ │   ║
║  │    ↓ (×0.5 + DropPath)                              │   ║
║  │  [Residual Connection]                              │   ║
║  │    ↓                                                 │   ║
║  │  ┌────────────────────────────────────────────────┐ │   ║
║  │  │ 2. Multi-Head Self-Attention                   │ │   ║
║  │  │    LayerNorm                                    │ │   ║
║  │  │    8 heads × 128 dim = 1024 total               │ │   ║
║  │  │    Attention(Q, K, V) with padding mask         │ │   ║
║  │  │    Dropout(0.3)                                 │ │   ║
║  │  └────────────────────────────────────────────────┘ │   ║
║  │    ↓ (+ DropPath)                                   │   ║
║  │  [Residual Connection]                              │   ║
║  │    ↓                                                 │   ║
║  │  ┌────────────────────────────────────────────────┐ │   ║
║  │  │ 3. Convolution Module                          │ │   ║
║  │  │    LayerNorm                                    │ │   ║
║  │  │    Pointwise Conv: 1024 → 2048                  │ │   ║
║  │  │    GLU (Gated Linear Unit): 2048 → 1024        │ │   ║
║  │  │    Depthwise Conv (k=31, groups=1024)          │ │   ║
║  │  │    LayerNorm                                    │ │   ║
║  │  │    SiLU activation                              │ │   ║
║  │  │    Pointwise Conv: 1024 → 1024                  │ │   ║
║  │  │    Dropout(0.3)                                 │ │   ║
║  │  └────────────────────────────────────────────────┘ │   ║
║  │    ↓                                                 │   ║
║  │  [Residual Connection]                              │   ║
║  │    ↓                                                 │   ║
║  │  ┌────────────────────────────────────────────────┐ │   ║
║  │  │ 4. Feed-Forward Module (half-step)             │ │   ║
║  │  │    LayerNorm                                    │ │   ║
║  │  │    Linear(1024 → 2048)                          │ │   ║
║  │  │    SiLU activation                              │ │   ║
║  │  │    Dropout(0.3)                                 │ │   ║
║  │  │    Linear(2048 → 1024)                          │ │   ║
║  │  │    Dropout(0.3)                                 │ │   ║
║  │  └────────────────────────────────────────────────┘ │   ║
║  │    ↓ (×0.5 + DropPath)                              │   ║
║  │  [Residual Connection]                              │   ║
║  │    ↓                                                 │   ║
║  │  Final LayerNorm                                    │   ║
║  │    ↓                                                 │   ║
║  │  Output [B, T/4, 1024]                              │   ║
║  └─────────────────────────────────────────────────────┘   ║
║                                                             ║
║  ┌─────────────────────────────────────────────────────┐   ║
║  │ INTERMEDIATE CTC (Layer 4, Training only)          │   ║
║  │ Linear(1024 → 41) → LogSoftmax                      │   ║
║  │ Loss weight: 0.3                                    │   ║
║  └─────────────────────────────────────────────────────┘   ║
╚═════════════════════════════════════════════════════════════╝
         |
         v [Batch, T/4, 1024]
┌─────────────────────────────────────────────────────────────┐
│  DEEP CLASSIFICATION HEAD (Linderman-style)                 │
│  Linear(1024 → 1024)                                        │
│    ↓                                                         │
│  LayerNorm                                                  │
│    ↓                                                         │
│  GELU activation                                            │
│    ↓                                                         │
│  Dropout(0.3)                                               │
│    ↓                                                         │
│  Linear(1024 → 41)  [40 phonemes + 1 blank]                │
│    ↓                                                         │
│  LogSoftmax                                                 │
└─────────────────────────────────────────────────────────────┘
         |
         v
OUTPUT: [T/4, Batch, 41] log-probabilities
         |
         v
┌─────────────────────────────────────────────────────────────┐
│  CTC DECODING                                               │
│  • Greedy decoding: argmax per timestep                     │
│  • Remove consecutive duplicates                            │
│  • Remove blank tokens (index 0)                            │
│  → Final phoneme sequence                                   │
└─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════
TRAINING DETAILS
═══════════════════════════════════════════════════════════════

Loss Function:
├─ Main CTC Loss (weight: 0.7)
│  └─ (1 - 0.1) × CTC + 0.1 × KL_div [label smoothing]
└─ Intermediate CTC Loss (weight: 0.3, layer 4)

Optimizer: AdamW
├─ Learning rate: 0.0004 (peak)
├─ Weight decay: 1e-3
├─ Warmup: 1000 steps
└─ Schedule: Cosine decay over 15,000 batches

Data Augmentation:
├─ White noise (σ=0.8)
├─ Constant offset (σ=0.2)
└─ SpecAugment (time + frequency masking)

Regularization:
├─ Dropout: 0.3
├─ Stochastic Depth (DropPath): 0.1
├─ Label Smoothing: 0.1
├─ Gradient Clipping: max_norm=1.0
└─ Weight Decay: 1e-3

═══════════════════════════════════════════════════════════════
KEY ARCHITECTURAL INNOVATIONS
═══════════════════════════════════════════════════════════════

1. Conformer Design:
   ✓ Combines global context (attention) + local patterns (conv)
   ✓ Better than pure Transformer or pure CNN for sequences

2. Day-Specific Normalization:
   ✓ Critical for multi-session BCI recordings
   ✓ Handles neural signal drift across days

3. Deep Supervision (InterCTC):
   ✓ Provides gradient signal to middle layers
   ✓ Improves training stability and convergence

4. Deep Classification Head:
   ✓ Non-linear 5-layer head vs simple linear projection
   ✓ Decouples feature learning from classification

5. SpecAugment:
   ✓ Adapted from speech recognition
   ✓ Forces model to use global context, not memorize

6. Frontend Temporal Processing:
   ✓ 4× downsampling reduces compute by ~16×
   ✓ Gaussian smoothing removes noise
   ✓ Matches GRU baseline preprocessing

═══════════════════════════════════════════════════════════════
MODEL STATISTICS
═══════════════════════════════════════════════════════════════

Parameters: ~45M (estimated)
Input: [Batch, ~1000 timesteps, 256 channels]
Output: [~242 timesteps, Batch, 41 classes]
Temporal compression: 4× (due to stride)
Performance: 16% CER (Character Error Rate)

Memory efficiency:
• Temporal downsampling: 4× fewer timesteps
• Mixed precision training support
• Gradient checkpointing compatible
```

## Information Flow Summary

```
Raw Neural Signal (256 channels, ~1kHz)
    ↓ [Day normalization]
    ↓ [Gaussian smoothing]
    ↓ [Temporal downsampling 4×]
Compressed Features (1024-dim, ~250Hz)
    ↓ [Autoencoder bottleneck]
    ↓ [SpecAugment masking]
    ↓ [Positional encoding]
Contextualized Features (8× Conformer blocks)
    ↓ [Global: Self-attention captures sentence structure]
    ↓ [Local: Convolution captures phoneme transitions]
    ↓ [Deep supervision at layer 4]
Phoneme Representations (1024-dim)
    ↓ [Deep non-linear classification head]
Phoneme Probabilities (41 classes)
    ↓ [CTC decoding]
Final Phoneme Sequence
```
