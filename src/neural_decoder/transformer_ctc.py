import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    """Stochastic Depth (DropPath) regularization."""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class DaySpecificLinear(nn.Module):
    """
    Per-day affine transform to handle cross-session variability.
    Mirrors the GRU baseline's dayWeights/dayBias mechanism.
    """

    def __init__(self, n_days: int, dim: int, init_identity: bool = True):
        super().__init__()
        self.dim = dim
        self.day_weights = nn.Parameter(torch.randn(n_days, dim, dim))
        self.day_bias = nn.Parameter(torch.zeros(n_days, 1, dim))
        if init_identity:
            with torch.no_grad():
                for d in range(n_days):
                    self.day_weights[d].copy_(torch.eye(dim))

    def forward(self, x: torch.Tensor, day_ids: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D], day_ids: [B]
        returns: [B, T, D]
        """
        day_w = torch.index_select(self.day_weights, 0, day_ids)
        day_b = torch.index_select(self.day_bias, 0, day_ids)
        return torch.einsum("btd,bdk->btk", x, day_w) + day_b


class NeuralFrontend(nn.Module):
    """
    Frontend processing: Gaussian smoothing → Strided temporal conv → Project
    """

    def __init__(
        self,
        n_channels: int,
        frontend_dim: int = 1024,
        dropout: float = 0.1,
        temporal_kernel: int = 32,
        temporal_stride: int = 4,
        gaussian_smooth_width: float = 2.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.temporal_kernel = temporal_kernel
        self.temporal_stride = temporal_stride

        # Gaussian smoothing (like GRU preprocessing)
        if gaussian_smooth_width > 0:
            kernel_size = int(gaussian_smooth_width * 4) + 1
            gaussian_kernel = self._make_gaussian_kernel(kernel_size, gaussian_smooth_width)
            self.register_buffer('gaussian_kernel', gaussian_kernel.view(1, 1, -1))
            self.gaussian_padding = kernel_size // 2
        else:
            self.gaussian_kernel = None

        # Strided temporal convolution (like GRU's unfold operation)
        if temporal_kernel > 0:
            self.temporal_conv = nn.Conv1d(
                n_channels, n_channels,
                kernel_size=temporal_kernel,
                stride=temporal_stride,
                padding=0,  # No padding to match GRU's unfold behavior
                groups=n_channels,
                bias=False
            )
            nn.init.constant_(self.temporal_conv.weight, 1.0 / temporal_kernel)
        else:
            self.temporal_conv = None

        # Project to frontend dimension
        self.proj = nn.Linear(n_channels, frontend_dim)
        self.ln = nn.LayerNorm(frontend_dim)
        self.dropout = nn.Dropout(dropout)

    def _make_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel for smoothing"""
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        returns: [B, T', C'] where T' ≈ T/stride
        """
        # Apply Gaussian smoothing first
        if self.gaussian_kernel is not None:
            x_t = x.transpose(1, 2)  # [B, C, T]
            kernel = self.gaussian_kernel.repeat(self.n_channels, 1, 1)
            x_smooth = F.conv1d(x_t, kernel, padding=self.gaussian_padding, groups=self.n_channels)
            x = x_smooth.transpose(1, 2)

        # Apply strided temporal convolution
        if self.temporal_conv is not None:
            x_t = x.transpose(1, 2)  # [B, C, T]
            x_strided = self.temporal_conv(x_t).transpose(1, 2)  # [B, T', C]
        else:
            x_strided = x

        # Project to frontend dimension
        x_proj = self.proj(x_strided)
        x_proj = self.ln(x_proj)
        x_proj = self.dropout(x_proj)
        return x_proj


class AutoEncoderEncoder(nn.Module):
    """
    MLP bottleneck projection (MiSTR-style).
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module: provides local pattern modeling.
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        # Pointwise conv (expansion)
        self.pw_conv1 = nn.Linear(d_model, d_model * 2)
        # GLU activation
        self.glu = nn.GLU(dim=-1)
        # Depthwise conv
        self.dw_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.ln_conv = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        # Pointwise conv (projection)
        self.pw_conv2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        residual = x
        x = self.ln(x)
        x = self.pw_conv1(x)  # [B, T, 2D]
        x = self.glu(x)  # [B, T, D]

        # Depthwise conv requires [B, D, T]
        x = x.transpose(1, 2)
        x = self.dw_conv(x)
        x = x.transpose(1, 2)

        x = self.ln_conv(x)
        x = self.activation(x)
        x = self.pw_conv2(x)
        x = self.dropout(x)
        return residual + x


class ConformerBlock(nn.Module):
    """
    Conformer block: FF + Attention + Conv + FF
    Half-step residual connections on feed-forward modules.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
        drop_path_prob: float = 0.1,
    ):
        super().__init__()
        # First feed-forward module (half-step residual)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Multi-head self-attention
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)

        # Convolution module
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)

        # Second feed-forward module (half-step residual)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.ln_final = nn.LayerNorm(d_model)

        # Stochastic Depth
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, D]
        src_key_padding_mask: [B, T] - True for padding positions
        returns: [B, T, D]
        """
        # First FF module (half-step) with DropPath
        x = x + self.drop_path(0.5 * self.ff1(x))

        # Multi-head attention with DropPath
        x_attn = self.ln_attn(x)
        attn_out, _ = self.attn(x_attn, x_attn, x_attn, key_padding_mask=src_key_padding_mask)
        x = x + self.drop_path(self.dropout_attn(attn_out))

        # Convolution module (already has residual inside)
        x = self.conv_module(x)

        # Second FF module (half-step) with DropPath
        x = x + self.drop_path(0.5 * self.ff2(x))

        x = self.ln_final(x)
        return x


class SpecAugment(nn.Module):
    """
    SpecAugment: Time and Feature Masking for speech data.
    Forces the model to rely on global context rather than local features.
    """

    def __init__(self, freq_mask_param: int = 27, time_mask_param: int = 35, num_freq_masks: int = 2, num_time_masks: int = 2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F] - Batch, Time, Features
        returns: [B, T, F] with random time/feature masks applied
        """
        if not self.training:
            return x

        B, T, F = x.shape

        # 1. Frequency (Feature) Masking
        # Mask random blocks of features
        for _ in range(self.num_freq_masks):
            f = int(torch.rand(1).item() * self.freq_mask_param)
            f = min(f, F)  # Ensure we don't exceed feature dimension
            if f > 0:
                f0 = int(torch.rand(1).item() * (F - f))
                x[:, :, f0:f0+f] = 0

        # 2. Time Masking
        # Mask random blocks of timesteps
        for _ in range(self.num_time_masks):
            t = int(torch.rand(1).item() * self.time_mask_param)
            t = min(t, T)  # Ensure we don't exceed time dimension
            if t > 0:
                t0 = int(torch.rand(1).item() * (T - t))
                x[:, t0:t0+t, :] = 0

        return x


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        return x + self.pe[:, : x.size(1)]


class NeuralTransformerCTCModel(nn.Module):
    """
    Conformer-based neural decoder for speech BCI.
    Architecture: Day Linear → Frontend → Autoencoder → Positional Encoding → Conformer → Output
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        n_days: int,
        frontend_dim: int = 1024,
        latent_dim: int = 1024,
        autoencoder_hidden_dim: int = 512,
        transformer_layers: int = 8,
        transformer_heads: int = 8,
        transformer_ff_dim: int = 2048,
        transformer_dropout: float = 0.3,
        temporal_kernel: int = 32,
        temporal_stride: int = 4,
        gaussian_smooth_width: float = 2.0,
        conformer_conv_kernel: int = 31,
        use_spec_augment: bool = True,
        spec_augment_freq_mask: int = 100,
        spec_augment_time_mask: int = 40,
        drop_path_prob: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.device_name = device

        # Day-specific normalization
        self.day_linear = DaySpecificLinear(n_days=n_days, dim=n_channels, init_identity=True)

        # Frontend: Gaussian smooth → Temporal conv/stride → Project
        self.frontend = NeuralFrontend(
            n_channels=n_channels,
            frontend_dim=frontend_dim,
            temporal_kernel=temporal_kernel,
            temporal_stride=temporal_stride,
            gaussian_smooth_width=gaussian_smooth_width,
            dropout=transformer_dropout,
        )

        # Autoencoder bottleneck
        self.encoder = AutoEncoderEncoder(
            input_dim=frontend_dim,
            latent_dim=latent_dim,
            hidden_dim=autoencoder_hidden_dim
        )

        # SpecAugment (applied after encoder, before positional encoding)
        self.use_spec_augment = use_spec_augment
        if use_spec_augment:
            self.spec_augment = SpecAugment(
                freq_mask_param=spec_augment_freq_mask,
                time_mask_param=spec_augment_time_mask,
                num_freq_masks=2,
                num_time_masks=2,
            )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model=latent_dim)

        # Conformer blocks
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                d_model=latent_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                conv_kernel_size=conformer_conv_kernel,
                drop_path_prob=drop_path_prob,
            )
            for _ in range(transformer_layers)
        ])

        # Intermediate CTC (for deep supervision)
        self.use_interctc = transformer_layers >= 6  # Only use if we have enough layers
        if self.use_interctc:
            # Apply intermediate loss at middle layer (e.g., layer 4 out of 8)
            self.interctc_layer = transformer_layers // 2
            self.inter_output = nn.Linear(latent_dim, n_classes)

        # Deep Classification Head (Linderman-style)
        # Decouples feature learning from classification for better performance
        self.output = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim, n_classes)
        )

        # Store temporal parameters for length computation
        self.temporal_kernel = temporal_kernel
        self.temporal_stride = temporal_stride

    def compute_output_lengths(self, input_lengths: torch.Tensor, actual_seq_len: int) -> torch.Tensor:
        """
        Compute output sequence lengths after temporal striding.
        Matches GRU formula: (length - kernel) / stride
        """
        if self.temporal_kernel > 0 and self.temporal_stride > 1:
            output_lengths = ((input_lengths - self.temporal_kernel) / self.temporal_stride).to(torch.int32)
        else:
            output_lengths = input_lengths
        return torch.clamp(output_lengths, max=actual_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        day_ids: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [B, T, C] - Neural features
        day_ids: [B] - Day indices
        input_lengths: [B] - Actual sequence lengths before padding

        returns:
            log_probs: [T', B, n_classes] - Log probabilities for CTC
            out_lengths: [B] - Actual output sequence lengths
            inter_log_probs: [T', B, n_classes] or None - Intermediate log probs for InterCTC
        """
        # Day-specific normalization
        x = self.day_linear(x, day_ids)

        # Frontend processing (includes striding)
        feats = self.frontend(x)

        # Autoencoder bottleneck
        z = self.encoder(feats)

        # Apply SpecAugment (only during training)
        if self.use_spec_augment:
            z = self.spec_augment(z)

        # Add positional encoding
        z = self.pos_enc(z)

        # Get actual sequence length after frontend processing
        actual_seq_len = z.size(1)

        # Create padding mask if input lengths provided
        padding_mask = None
        if input_lengths is not None:
            out_lengths = self.compute_output_lengths(input_lengths, actual_seq_len)
            max_len = z.size(1)
            mask = torch.arange(max_len, device=z.device).expand(len(out_lengths), max_len) >= out_lengths.unsqueeze(1)
            padding_mask = mask  # [B, T]
        else:
            out_lengths = torch.full((x.size(0),), actual_seq_len, dtype=torch.int32, device=x.device)

        # Apply Conformer blocks with intermediate CTC
        inter_log_probs = None
        for i, layer in enumerate(self.conformer_layers):
            z = layer(z, src_key_padding_mask=padding_mask)

            # Capture intermediate output for InterCTC (only during training)
            if self.use_interctc and i == self.interctc_layer - 1 and self.training:
                inter_logits = self.inter_output(z)  # [B, T, C]
                inter_log_probs = inter_logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, C]

        # Output projection and log softmax
        logits = self.output(z)  # [B, T, C]
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, C] for CTC

        return log_probs, out_lengths, inter_log_probs
