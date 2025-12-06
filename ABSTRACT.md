# Abstract: Conformer-based Neural Decoder for Intracortical Speech Brain-Computer Interfaces

We present a transformer-based neural decoder for decoding speech from intracortical recordings in brain-computer interfaces (BCIs). Our approach adapts the Conformer architecture—originally developed for automatic speech recognition—to directly decode phoneme sequences from multi-channel neural activity. The model processes 256-channel neural recordings through a multi-stage pipeline: day-specific linear transforms handle cross-session variability, a strided temporal convolutional frontend performs 4× downsampling, and an autoencoder bottleneck enforces information compression. Eight Conformer blocks combine global context modeling via multi-head self-attention with local pattern extraction through depthwise convolution (kernel size 31), effectively capturing both sentence-level structure and phoneme-level transitions. We employ extensive regularization including SpecAugment (time and frequency masking), stochastic depth, label smoothing, and intermediate CTC loss at the fourth layer for deep supervision. A deep non-linear classification head decouples feature learning from phoneme classification. Training uses Connectionist Temporal Classification (CTC) loss to handle variable-length alignment between neural activity and phoneme sequences. On a speech BCI dataset, our model achieves 16% character error rate, demonstrating that modern transformer architectures can effectively decode speech from neural recordings when augmented with domain-specific adaptations for cross-session robustness and temporal processing.

---

## Key Contributions

1. **First application of Conformer architecture to intracortical speech BCI**, demonstrating superior performance over recurrent baselines through combined attention and convolution mechanisms.

2. **Domain-specific adaptations for neural recordings**: day-specific normalization for cross-session drift, Gaussian smoothing for noise reduction, and temporal downsampling for computational efficiency.

3. **Comprehensive regularization strategy**: SpecAugment adapted from speech recognition, intermediate CTC for deep supervision, and stochastic depth for improved gradient flow.

4. **State-of-the-art performance**: 16% CER achieved through synergistic combination of modern deep learning techniques and BCI-specific preprocessing.
