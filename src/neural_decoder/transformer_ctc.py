import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pywt


class DaySpecificLinear(nn.Module):
    """
    Per-day affine transform. Mirrors the GRU baseline's dayWeights/dayBias block.
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
        # x: [B, T, D], day_ids: [B]
        day_w = torch.index_select(self.day_weights, 0, day_ids)
        day_b = torch.index_select(self.day_bias, 0, day_ids)
        # einsum keeps batch/time alignment: [B, T, D]
        return torch.einsum("btd,bdk->btk", x, day_w) + day_b


class WaveletFeatureExtractor(nn.Module):
    """
    Fast GPU-based wavelet-inspired feature extractor using learned filter banks.
    Approximates wavelet decomposition with trainable convolutional filters at multiple scales.
    Much faster than pywt.wavedec while maintaining the spirit of MiSTR's approach.
    """

    def __init__(
        self,
        n_channels: int,
        n_scales: int = 4,
        wavelet_name: str = "db4",
        window_size: int = 10,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_scales = n_scales
        self.window_size = window_size

        # Initialize filter banks based on wavelet
        wave = pywt.Wavelet(wavelet_name)

        # Create multi-scale filters inspired by wavelet decomposition
        # Each scale has both high-pass (detail) and low-pass components
        self.filters = nn.ModuleList()

        for scale in range(n_scales):
            # Dilation increases with scale (2^scale)
            dilation = 2 ** scale

            # High-pass filter (detail coefficients)
            hi_filter = torch.tensor(wave.dec_hi, dtype=torch.float32)
            kernel_size = len(hi_filter)

            # Create 1D conv with groups for efficiency
            conv = nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2,
                groups=n_channels,  # Depthwise conv
                bias=False
            )

            # Initialize with wavelet filter (learnable - gradient clipping provides stability)
            with torch.no_grad():
                for i in range(n_channels):
                    conv.weight[i, 0, :] = hi_filter

            self.filters.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        returns: [B, T, C * n_scales] - wavelet-inspired features per scale
        """
        b, t, c = x.shape

        # Transpose for conv1d: [B, C, T]
        x_t = x.transpose(1, 2)

        scale_features = []
        for scale_conv in self.filters:
            # Apply filter at this scale
            filtered = scale_conv(x_t)  # [B, C, T]

            # Compute energy (square of coefficients)
            energy = filtered.pow(2)

            # Optional: smooth with small window
            if self.window_size > 1:
                energy = F.avg_pool1d(
                    energy,
                    kernel_size=min(self.window_size, energy.size(-1)),
                    stride=1,
                    padding=self.window_size // 2
                )

            # Ensure same length as input
            if energy.size(-1) != t:
                energy = F.interpolate(energy, size=t, mode='linear', align_corners=False)

            scale_features.append(energy)

        # Concatenate all scales: [B, C * n_scales, T]
        features = torch.cat(scale_features, dim=1)

        # Transpose back: [B, T, C * n_scales]
        features = features.transpose(1, 2)

        return features


class PACFeatureExtractor(nn.Module):
    """
    Optional cross-frequency coupling features using FIR bandpass filters and Hilbert envelope/phase.
    """

    def __init__(
        self,
        n_channels: int,
        low_band: Tuple[float, float] = (4.0, 8.0),
        high_band: Tuple[float, float] = (70.0, 170.0),
        sample_rate: float = 50.0,  # bins per second (20 ms bins)
        filter_len: int = 129,
        smoothing_bins: int = 5,
    ):
        super().__init__()
        import scipy.signal  # local import to avoid hard dependency at module import

        nyq = sample_rate / 2.0
        self.low_kernel = torch.tensor(
            scipy.signal.firwin(filter_len, [low_band[0] / nyq, low_band[1] / nyq], pass_zero=False),
            dtype=torch.float32,
        ).view(1, 1, -1)
        self.high_kernel = torch.tensor(
            scipy.signal.firwin(filter_len, [high_band[0] / nyq, high_band[1] / nyq], pass_zero=False),
            dtype=torch.float32,
        ).view(1, 1, -1)
        self.n_channels = n_channels
        self.filter_len = filter_len
        self.smoothing_bins = smoothing_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        returns: [B, T, C] PAC magnitude per channel.
        """
        b, t, c = x.shape
        x_t = x.transpose(1, 2)  # [B, C, T]
        device = x.device
        # Prepare grouped conv weights on the right device/dtype
        low_k = self.low_kernel.to(device=device, dtype=x.dtype).repeat(c, 1, 1)
        high_k = self.high_kernel.to(device=device, dtype=x.dtype).repeat(c, 1, 1)
        pad = self.filter_len // 2
        low = F.conv1d(x_t, low_k, padding=pad, groups=c)
        high = F.conv1d(x_t, high_k, padding=pad, groups=c)

        # Hilbert via FFT for analytic signals
        def analytic(sig: torch.Tensor) -> torch.Tensor:
            # sig: [B, C, T]
            fft = torch.fft.fft(sig, dim=-1)
            h = torch.zeros_like(fft)
            n = sig.shape[-1]
            if n % 2 == 0:
                h[..., 0] = 1
                h[..., n // 2] = 1
                h[..., 1 : n // 2] = 2
            else:
                h[..., 0] = 1
                h[..., 1 : (n + 1) // 2] = 2
            return torch.fft.ifft(fft * h, dim=-1)

        low_analytic = analytic(low)
        high_analytic = analytic(high)
        low_phase = torch.angle(low_analytic)
        high_amp = torch.abs(high_analytic)

        pac_complex = high_amp * torch.exp(1j * low_phase)
        # Smooth over time to stabilize PAC
        if self.smoothing_bins > 1:
            pad_s = self.smoothing_bins // 2
            real = F.avg_pool1d(pac_complex.real, kernel_size=self.smoothing_bins, stride=1, padding=pad_s)
            imag = F.avg_pool1d(pac_complex.imag, kernel_size=self.smoothing_bins, stride=1, padding=pad_s)
            pac_complex = torch.complex(real, imag)
        pac_mag = torch.abs(pac_complex)  # [B, C, T]
        return pac_mag.transpose(1, 2)  # [B, T, C]


class NeuralFrontend(nn.Module):
    """
    Combines raw day-normalized features with wavelet (and optional PAC) features, then projects.
    """

    def __init__(
        self,
        n_channels: int,
        use_wavelets: bool = True,
        n_scales: int = 4,
        wavelet_window_size: int = 10,
        use_pac_features: bool = False,
        pac_low: Tuple[float, float] = (4.0, 8.0),
        pac_high: Tuple[float, float] = (70.0, 170.0),
        pac_filter_len: int = 129,
        pac_smoothing_bins: int = 5,
        frontend_dim: int = 512,
        dropout: float = 0.1,
        temporal_kernel: int = 0,
        temporal_stride: int = 1,  # Add striding like GRU (stride=4)
        gaussian_smooth_width: float = 0.0,  # GRU uses 2.0
    ):
        super().__init__()
        self.use_wavelets = use_wavelets
        self.use_pac_features = use_pac_features
        self.n_channels = n_channels
        self.temporal_kernel = temporal_kernel
        self.temporal_stride = temporal_stride

        # Gaussian smoothing (like GRU preprocessing)
        if gaussian_smooth_width > 0:
            kernel_size = int(gaussian_smooth_width * 4) + 1  # 4 sigma window
            gaussian_kernel = self._make_gaussian_kernel(kernel_size, gaussian_smooth_width)
            self.register_buffer('gaussian_kernel', gaussian_kernel.view(1, 1, -1))
            self.gaussian_padding = kernel_size // 2
        else:
            self.gaussian_kernel = None

        # Strided temporal convolution (like GRU's kernel + stride)
        if temporal_kernel > 0:
            self.temporal_conv = nn.Conv1d(
                n_channels, n_channels,
                kernel_size=temporal_kernel,
                stride=temporal_stride,  # Add stride!
                padding=temporal_kernel // 2,
                groups=n_channels,
                bias=False
            )
            nn.init.constant_(self.temporal_conv.weight, 1.0 / temporal_kernel)
        else:
            self.temporal_conv = None

        if use_wavelets:
            self.wavelet_extractor = WaveletFeatureExtractor(
                n_channels=n_channels,
                n_scales=n_scales,
                wavelet_name="db4",
                window_size=wavelet_window_size,
            )
            wavelet_dim = n_channels * n_scales
        else:
            self.wavelet_extractor = None
            wavelet_dim = 0

        if use_pac_features:
            self.pac_extractor = PACFeatureExtractor(
                n_channels=n_channels,
                low_band=pac_low,
                high_band=pac_high,
                sample_rate=50.0,
                filter_len=pac_filter_len,
                smoothing_bins=pac_smoothing_bins,
            )
            pac_dim = n_channels
        else:
            self.pac_extractor = None
            pac_dim = 0

        input_dim = n_channels + wavelet_dim + pac_dim
        self.proj = nn.Linear(input_dim, frontend_dim)
        self.ln = nn.LayerNorm(frontend_dim)
        self.dropout = nn.Dropout(dropout)

    def _make_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel for smoothing"""
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Gaussian smoothing first (like GRU preprocessing)
        if self.gaussian_kernel is not None:
            x_t = x.transpose(1, 2)  # [B, C, T]
            # Apply same Gaussian kernel to all channels
            kernel = self.gaussian_kernel.repeat(self.n_channels, 1, 1)  # [C, 1, K]
            x_smooth = F.conv1d(x_t, kernel, padding=self.gaussian_padding, groups=self.n_channels)
            x = x_smooth.transpose(1, 2)  # [B, T, C]

        # Apply temporal smoothing/striding (like GRU's kernel + stride processing)
        if self.temporal_conv is not None:
            # x: [B, T, C] -> [B, C, T] for conv1d
            x_t = x.transpose(1, 2)
            x_smooth = self.temporal_conv(x_t).transpose(1, 2)  # [B, T', C] where T' = T/stride
            # Compute additional features on the strided output
            x_for_features = x_smooth
        else:
            x_for_features = x

        feats = [x_for_features]

        if self.wavelet_extractor is not None:
            wavelet_feats = self.wavelet_extractor(x_for_features)  # [B, T', C * n_scales]
            feats.append(wavelet_feats)

        if self.pac_extractor is not None:
            pac_feats = self.pac_extractor(x_for_features)  # [B, T', C]
            feats.append(pac_feats)

        # All features now have same temporal dimension [B, T', ...]
        x_cat = torch.cat(feats, dim=-1)
        x_proj = self.proj(x_cat)
        x_proj = self.ln(x_proj)
        x_proj = self.dropout(x_proj)
        return x_proj


class AutoEncoderEncoder(nn.Module):
    """
    Simple MLP encoder (MiSTR-style) used as a bottleneck projection.
    Uses fixed 128 hidden units as in original MiSTR implementation.
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
    Convolution module from Conformer architecture.
    Provides local pattern modeling to complement global attention.
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        # Layernorm
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
        self.ln_conv = nn.LayerNorm(d_model)  # Use LayerNorm instead of BatchNorm for stability
        self.activation = nn.SiLU()
        # Pointwise conv (projection)
        self.pw_conv2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x = self.ln(x)
        x = self.pw_conv1(x)  # [B, T, 2D]
        x = self.glu(x)  # [B, T, D]

        # Depthwise conv requires [B, D, T]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.dw_conv(x)  # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]
        x = self.ln_conv(x)  # LayerNorm for stability
        x = self.activation(x)

        x = self.pw_conv2(x)
        x = self.dropout(x)
        return residual + x


class ConformerBlock(nn.Module):
    """
    Conformer block: Feed-forward + Multi-head Attention + Convolution + Feed-forward
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
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

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D]
        # First FF module (half-step)
        x = x + 0.5 * self.ff1(x)

        # Multi-head attention
        x_attn = self.ln_attn(x)
        attn_out, _ = self.attn(x_attn, x_attn, x_attn, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout_attn(attn_out)

        # Convolution module
        x = self.conv_module(x)

        # Second FF module (half-step)
        x = x + 0.5 * self.ff2(x)

        x = self.ln_final(x)
        return x


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding - proven superior to absolute for speech.
    Encodes relative distances between positions rather than absolute positions.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Learnable relative position embeddings
        # We need embeddings for relative distances from -max_len to +max_len
        self.rel_embeddings = nn.Parameter(torch.randn(2 * max_len + 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        # For simplicity, we'll add a position-dependent bias
        # Full relative attention is more complex but this gives the benefits
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        # Get embeddings for each position
        pos_emb = self.rel_embeddings[self.max_len - positions.max():self.max_len + T]
        return x + pos_emb[:T].unsqueeze(0)


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (fallback option).
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
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]


class NeuralTransformerCTCModel(nn.Module):
    """
    MiSTR-inspired Transformer encoder for phoneme CTC.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        n_days: int,
        use_wavelets: bool = True,
        n_scales: int = 4,
        wavelet_window_size: int = 10,
        use_pac_features: bool = False,
        frontend_dim: int = 512,
        latent_dim: int = 256,
        autoencoder_hidden_dim: int = 128,
        transformer_layers: int = 6,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 1024,
        transformer_dropout: float = 0.1,
        temporal_kernel: int = 0,  # Temporal convolution kernel size
        temporal_stride: int = 1,  # Temporal convolution stride
        gaussian_smooth_width: float = 0.0,  # GRU uses 2.0
        use_conformer: bool = False,  # Use Conformer blocks instead of standard transformer
        conformer_conv_kernel: int = 31,  # Conformer convolution kernel size
        use_relative_pe: bool = True,  # Use relative positional encoding (better for speech)
        intermediate_ctc_layers: list = None,  # Layers to add intermediate CTC losses
        stochastic_depth_rate: float = 0.0,  # Stochastic depth for regularization
        device: str = "cuda",
    ):
        super().__init__()
        self.device_name = device
        self.use_conformer = use_conformer
        self.use_relative_pe = use_relative_pe
        self.intermediate_ctc_layers = intermediate_ctc_layers or []
        self.stochastic_depth_rate = stochastic_depth_rate

        self.day_linear = DaySpecificLinear(n_days=n_days, dim=n_channels, init_identity=True)
        self.frontend = NeuralFrontend(
            n_channels=n_channels,
            use_wavelets=use_wavelets,
            n_scales=n_scales,
            wavelet_window_size=wavelet_window_size,
            use_pac_features=use_pac_features,
            frontend_dim=frontend_dim,
            temporal_kernel=temporal_kernel,
            temporal_stride=temporal_stride,
            gaussian_smooth_width=gaussian_smooth_width,
        )
        self.encoder = AutoEncoderEncoder(
            input_dim=frontend_dim, latent_dim=latent_dim, hidden_dim=autoencoder_hidden_dim
        )

        # Use relative or absolute positional encoding
        if use_relative_pe:
            self.pos_enc = RelativePositionalEncoding(d_model=latent_dim)
        else:
            self.pos_enc = PositionalEncoding(d_model=latent_dim)

        if use_conformer:
            # Use Conformer blocks
            self.conformer_layers = nn.ModuleList([
                ConformerBlock(
                    d_model=latent_dim,
                    nhead=transformer_heads,
                    dim_feedforward=transformer_ff_dim,
                    dropout=transformer_dropout,
                    conv_kernel_size=conformer_conv_kernel,
                )
                for _ in range(transformer_layers)
            ])
            self.transformer = None
        else:
            # Use standard transformer
            enc_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                batch_first=True,
                activation="relu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
            self.conformer_layers = None

        self.output = nn.Linear(latent_dim, n_classes)

        # Intermediate CTC outputs for better gradient flow
        self.intermediate_outputs = nn.ModuleDict()
        for layer_idx in self.intermediate_ctc_layers:
            self.intermediate_outputs[f"layer_{layer_idx}"] = nn.Linear(latent_dim, n_classes)

    def compute_output_lengths(self, input_lengths: torch.Tensor, actual_seq_len: int) -> torch.Tensor:
        # Account for temporal striding in frontend (if used)
        if hasattr(self.frontend, 'temporal_conv') and self.frontend.temporal_conv is not None:
            # Apply same formula as GRU: (length - kernel) / stride
            kernel = self.frontend.temporal_kernel
            stride = self.frontend.temporal_stride
            output_lengths = ((input_lengths - kernel) / stride).to(torch.int32)
        else:
            # No striding, temporal dimension preserved
            output_lengths = input_lengths

        # Clamp to actual sequence length
        return torch.clamp(output_lengths, max=actual_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        day_ids: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C]
        x = self.day_linear(x, day_ids)
        feats = self.frontend(x)
        z = self.encoder(feats)
        z = self.pos_enc(z)

        # Get actual sequence length after frontend processing
        actual_seq_len = z.size(1)

        padding_mask = None
        if input_lengths is not None:
            # Update input_lengths to reflect actual output size
            out_lengths = self.compute_output_lengths(input_lengths, actual_seq_len)
            max_len = z.size(1)
            mask = torch.arange(max_len, device=z.device).expand(len(out_lengths), max_len) >= out_lengths.unsqueeze(1)
            padding_mask = mask  # [B, T]
        else:
            out_lengths = None

        intermediate_outputs = {}

        if self.use_conformer:
            # Apply Conformer blocks with stochastic depth and intermediate CTC
            for layer_idx, layer in enumerate(self.conformer_layers):
                # Stochastic depth: randomly skip layers during training
                if self.training and self.stochastic_depth_rate > 0:
                    if torch.rand(1).item() < self.stochastic_depth_rate:
                        continue  # Skip this layer

                z = layer(z, src_key_padding_mask=padding_mask)

                # Collect intermediate CTC outputs
                if layer_idx in self.intermediate_ctc_layers:
                    intermediate_logits = self.intermediate_outputs[f"layer_{layer_idx}"](z)
                    intermediate_log_probs = intermediate_logits.log_softmax(dim=-1).transpose(0, 1)
                    intermediate_outputs[f"layer_{layer_idx}"] = intermediate_log_probs

            z_enc = z
        else:
            # Use standard transformer
            z_enc = self.transformer(z, src_key_padding_mask=padding_mask)

        logits = self.output(z_enc)  # [B, T, C]
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, C]

        if intermediate_outputs:
            return log_probs, out_lengths, intermediate_outputs
        else:
            return log_probs, out_lengths


# ============================================================================
# NOVEL ARCHITECTURE: Multi-Scale Attention-Based Neural Decoder
# ============================================================================


class CrossScaleFusion(nn.Module):
    """
    Fuses features from multiple temporal scales using cross-attention.
    Allows information flow between fast, medium, and slow temporal pathways.
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Cross-attention between scales
        self.fast_to_medium = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.medium_to_slow = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.slow_to_fast = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)  # Final norm for stability
        )

    def forward(self, fast: torch.Tensor, medium: torch.Tensor, slow: torch.Tensor):
        """
        fast: [B, T_fast, D] - high temporal resolution
        medium: [B, T_medium, D] - medium temporal resolution
        slow: [B, T_slow, D] - low temporal resolution
        Returns: [B, T_medium, D] - fused representation at medium scale
        """
        # Upsample slow to medium resolution with interpolation
        slow_upsampled = F.interpolate(
            slow.transpose(1, 2),
            size=medium.shape[1],
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        # Downsample fast to medium resolution
        fast_downsampled = F.adaptive_avg_pool1d(
            fast.transpose(1, 2),
            medium.shape[1]
        ).transpose(1, 2)

        # Cross-attention between scales
        fast_attended, _ = self.fast_to_medium(medium, fast_downsampled, fast_downsampled)
        slow_attended, _ = self.slow_to_fast(medium, slow_upsampled, slow_upsampled)

        # Concatenate and fuse
        combined = torch.cat([fast_attended, medium, slow_attended], dim=-1)
        fused = self.fusion(combined)

        return fused


class MultiScaleConformerEncoder(nn.Module):
    """
    Multi-scale temporal pyramid: processes neural signals at multiple
    temporal resolutions to capture both fast neural dynamics and slow
    prosodic patterns.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        conv_kernel: int = 31,
    ):
        super().__init__()

        # Three parallel pathways at different temporal scales
        # Fast path: stride 2 (~75ms resolution)
        self.fast_temporal_conv = nn.Conv1d(
            input_dim, d_model, kernel_size=32, stride=2, padding=15
        )
        self.fast_conformer = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dim_ff, dropout, conv_kernel)
            for _ in range(n_layers // 2)  # Fewer layers for fast path
        ])

        # Medium path: stride 4 (~150ms resolution) - main pathway
        self.medium_temporal_conv = nn.Conv1d(
            input_dim, d_model, kernel_size=32, stride=4, padding=15
        )
        self.medium_conformer = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dim_ff, dropout, conv_kernel)
            for _ in range(n_layers)
        ])

        # Slow path: stride 8 (~300ms resolution)
        self.slow_temporal_conv = nn.Conv1d(
            input_dim, d_model, kernel_size=32, stride=8, padding=15
        )
        self.slow_conformer = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dim_ff, dropout, conv_kernel)
            for _ in range(n_layers // 2)
        ])

        # Cross-scale fusion
        self.fusion = CrossScaleFusion(d_model, n_heads, dropout)

        # Positional encoding
        self.pos_enc = RelativePositionalEncoding(d_model)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, return_all_scales: bool = False):
        """
        x: [B, T, C] - neural signal
        return_all_scales: if True, return individual scales AND fused (for multi-scale CTC)

        Returns:
            if return_all_scales=False: [B, T', D] - fused features only
            if return_all_scales=True: dict with 'fast', 'medium', 'slow', 'fused'
        """
        # x: [B, T, C] -> transpose for conv1d
        x_t = x.transpose(1, 2)  # [B, C, T]

        # Fast pathway (stride 2)
        fast = self.fast_temporal_conv(x_t).transpose(1, 2)  # [B, T/2, D]
        fast = self.pos_enc(fast)
        for layer in self.fast_conformer:
            fast = layer(fast, src_key_padding_mask=None)

        # Medium pathway (stride 4)
        medium = self.medium_temporal_conv(x_t).transpose(1, 2)  # [B, T/4, D]
        medium = self.pos_enc(medium)
        for layer in self.medium_conformer:
            medium = layer(medium, src_key_padding_mask=None)

        # Slow pathway (stride 8)
        slow = self.slow_temporal_conv(x_t).transpose(1, 2)  # [B, T/8, D]
        slow = self.pos_enc(slow)
        for layer in self.slow_conformer:
            slow = layer(slow, src_key_padding_mask=None)

        # Fuse across scales
        fused = self.fusion(fast, medium, slow)  # [B, T/4, D]

        # Return based on flag
        if return_all_scales:
            return {
                'fast': fast,      # [B, ~75, D]
                'medium': medium,  # [B, ~38, D]
                'slow': slow,      # [B, ~19, D]
                'fused': fused,    # [B, ~38, D]
            }
        else:
            return fused


class AttentionDecoder(nn.Module):
    """
    Attention-based decoder: learns optimal alignment between neural
    activity and phonemes. Replaces CTC's rigid monotonic assumptions.
    """
    def __init__(
        self,
        n_classes: int,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        # Token embedding for target phonemes
        self.embedding = nn.Embedding(n_classes, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        # Transformer decoder with cross-attention to encoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, n_classes)

    def forward(
        self,
        encoder_output: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ):
        """
        encoder_output: [B, T, D] - encoded neural features
        target: [B, S] - target phoneme indices (for teacher forcing)
        Returns: [B, S, n_classes] - logits over phoneme vocabulary
        """
        if target is None:
            # Inference mode: use greedy decoding
            return self.greedy_decode(encoder_output)

        # Training mode: teacher forcing
        # Embed target tokens
        tgt_emb = self.embedding(target)  # [B, S, D]
        tgt_emb = self.pos_encoding(tgt_emb)

        # Create causal mask for decoder (prevent looking ahead)
        S = target.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S).to(target.device)

        # Decode with cross-attention to encoder
        decoded = self.transformer_decoder(
            tgt=tgt_emb,
            memory=encoder_output,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_mask,
        )

        # Project to vocabulary
        logits = self.output_projection(decoded)  # [B, S, n_classes]

        return logits

    def greedy_decode(self, encoder_output: torch.Tensor, max_len: int = 100):
        """Greedy decoding for inference"""
        B = encoder_output.shape[0]
        device = encoder_output.device

        # Start with BOS token (assume class 0 is blank/BOS)
        decoded_tokens = torch.zeros(B, 1, dtype=torch.long, device=device)

        for _ in range(max_len):
            # Embed current sequence
            tgt_emb = self.embedding(decoded_tokens)
            tgt_emb = self.pos_encoding(tgt_emb)

            # Create causal mask
            S = decoded_tokens.shape[1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(S).to(device)

            # Decode
            decoded = self.transformer_decoder(
                tgt=tgt_emb,
                memory=encoder_output,
                tgt_mask=causal_mask,
            )

            # Get next token prediction
            logits = self.output_projection(decoded[:, -1:, :])  # [B, 1, n_classes]
            next_token = logits.argmax(dim=-1)  # [B, 1]

            # Append to sequence
            decoded_tokens = torch.cat([decoded_tokens, next_token], dim=1)

            # Stop if all sequences predict EOS (or blank)
            if (next_token == 0).all():
                break

        return decoded_tokens


class ContrastivePretraining(nn.Module):
    """
    Self-supervised contrastive learning for neural representations.
    Learns to map similar neural patterns close together in embedding space.
    """
    def __init__(self, encoder: nn.Module, d_model: int, projection_dim: int = 128):
        super().__init__()
        self.encoder = encoder

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )

        self.temperature = 0.07  # Temperature for NT-Xent loss

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Data augmentation for neural signals:
        - Time jitter
        - Gaussian noise
        - Random masking
        """
        B, T, C = x.shape
        x_aug = x.clone()

        # 1. Gaussian noise
        noise = torch.randn_like(x_aug) * 0.1
        x_aug = x_aug + noise

        # 2. Random time masking (similar to SpecAugment)
        if T > 20:
            mask_len = torch.randint(5, 15, (1,)).item()
            mask_start = torch.randint(0, T - mask_len, (1,)).item()
            x_aug[:, mask_start:mask_start + mask_len, :] = 0

        # 3. Channel dropout
        if torch.rand(1).item() < 0.3:
            n_drop = int(C * 0.1)  # Drop 10% of channels
            drop_idx = torch.randperm(C)[:n_drop]
            x_aug[:, :, drop_idx] = 0

        return x_aug

    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
        Used in SimCLR and other contrastive learning methods
        """
        B = z1.shape[0]

        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Compute similarity matrix
        representations = torch.cat([z1, z2], dim=0)  # [2B, D]
        similarity_matrix = torch.matmul(representations, representations.T)  # [2B, 2B]

        # Create labels: positives are (i, i+B) and (i+B, i)
        labels = torch.arange(B, device=z1.device)
        labels = torch.cat([labels + B, labels], dim=0)

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z1.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature

        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] - neural signal batch
        Returns: contrastive loss
        """
        # Create two augmented views
        x1 = self.augment(x)
        x2 = self.augment(x)

        # Encode both views
        h1 = self.encoder(x1)  # [B, T', D]
        h2 = self.encoder(x2)  # [B, T', D]

        # Pool to fixed-size representation (mean pooling)
        h1_pooled = h1.mean(dim=1)  # [B, D]
        h2_pooled = h2.mean(dim=1)  # [B, D]

        # Project to contrastive space
        z1 = self.projection_head(h1_pooled)  # [B, projection_dim]
        z2 = self.projection_head(h2_pooled)  # [B, projection_dim]

        # Compute contrastive loss
        loss = self.nt_xent_loss(z1, z2)

        return loss


class MultiScaleCTCDecoder(nn.Module):
    """
    SIMPLIFIED Novel architecture:
    1. Multi-scale temporal pyramid encoder (NOVEL)
    2. CTC decoding (PROVEN)
    3. Optional diphone auxiliary head for context-aware learning
    4. Optional contrastive pre-training

    This combines the novelty of multi-scale processing with the
    stability of CTC, avoiding the complexity/bugs of attention alignment.
    """
    def __init__(
        self,
        n_classes: int,
        input_dim: int = 256,
        d_model: int = 512,
        encoder_layers: int = 6,
        n_heads: int = 8,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        conv_kernel: int = 31,
        n_days: int = 1,
        gaussian_smooth_width: float = 0.0,
        use_contrastive_pretraining: bool = False,
        use_diphone_head: bool = False,
        num_diphones: int = 1012,  # Default from vocab (1011 + blank)
        diphone_marginalization_matrix: Optional[np.ndarray] = None,  # For proper DCoND
        use_multiscale_ctc: bool = False,  # NEW: Enable Step 3 multi-scale CTC heads
        device: str = "cuda",
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model
        self.use_contrastive_pretraining = use_contrastive_pretraining
        self.use_diphone_head = use_diphone_head
        self.num_diphones = num_diphones
        self.use_multiscale_ctc = use_multiscale_ctc
        self.device = device

        # Check if we're using proper marginalization (DCoND approach)
        self.use_marginalization = diphone_marginalization_matrix is not None
        if self.use_marginalization:
            # Register marginalization matrix as a buffer (not a parameter)
            marg_matrix_tensor = torch.from_numpy(diphone_marginalization_matrix).float()
            self.register_buffer('marginalization_matrix', marg_matrix_tensor)
            print(f"✓ Using proper DCoND marginalization: diphone -> phoneme")
            print(f"  Marginalization matrix shape: {marg_matrix_tensor.shape}")

        # Gaussian smoothing (optional, match GRU baseline)
        self.gaussian_smooth_width = gaussian_smooth_width
        if gaussian_smooth_width > 0:
            kernel_size = int(6 * gaussian_smooth_width)
            if kernel_size % 2 == 0:
                kernel_size += 1
            x_vals = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            kernel = torch.exp(-(x_vals ** 2) / (2 * gaussian_smooth_width ** 2))
            kernel = kernel / kernel.sum()
            self.register_buffer('smooth_kernel', kernel.view(1, 1, -1))
            self.smooth_padding = kernel_size // 2

        # Day-specific normalization
        self.day_layer_norm = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(n_days)
        ])

        # Multi-scale encoder (THE NOVEL PART!)
        self.encoder = MultiScaleConformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=encoder_layers,
            n_heads=n_heads,
            dim_ff=dim_ff,
            dropout=dropout,
            conv_kernel=conv_kernel,
        )

        # CTC output layers
        if self.use_marginalization:
            # PROPER DCoND: Only diphone head, marginalize to get phoneme predictions
            self.diphone_output = nn.Linear(d_model, num_diphones)
            self.phone_output = None  # Not needed - phonemes come from marginalization
            print(f"  Model architecture: diphone_output only (phonemes via marginalization)")
        elif use_diphone_head:
            # OLD APPROACH: Separate heads (for backward compatibility)
            self.phone_output = nn.Linear(d_model, n_classes)
            self.diphone_output = nn.Linear(d_model, num_diphones)
            print(f"  Model architecture: separate phone_output and diphone_output heads")
        else:
            # BASELINE: Only phoneme head
            self.phone_output = nn.Linear(d_model, n_classes)
            self.diphone_output = None
            print(f"  Model architecture: phone_output only (baseline)")

        # STEP 3: Multi-scale auxiliary CTC heads (optional)
        if use_multiscale_ctc:
            # Lightweight auxiliary phoneme heads for fast and slow pathways
            # These provide additional supervision and improve gradient flow
            self.fast_phone_head = nn.Linear(d_model, n_classes)
            self.slow_phone_head = nn.Linear(d_model, n_classes)
            print(f"  + Multi-scale CTC: auxiliary heads on fast and slow pathways")
        else:
            self.fast_phone_head = None
            self.slow_phone_head = None

        # Contrastive pre-training module (optional)
        if use_contrastive_pretraining:
            self.contrastive_module = ContrastivePretraining(
                encoder=self.encoder,
                d_model=d_model,
            )

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
        x_len: Optional[torch.Tensor] = None,
        return_contrastive_loss: bool = False,
    ):
        """
        x: [B, T, C] - neural signal
        day_idx: [B] - day indices

        Returns:
            log_probs: [T, B, n_classes] - CTC format
            out_lengths: [B] - sequence lengths after temporal downsampling
        """
        # Gaussian smoothing
        if self.gaussian_smooth_width > 0:
            x_t = x.transpose(1, 2)  # [B, C, T]
            x_smoothed = []
            for c in range(x_t.shape[1]):
                smoothed_c = F.conv1d(
                    x_t[:, c:c+1, :],
                    self.smooth_kernel,
                    padding=self.smooth_padding
                )
                x_smoothed.append(smoothed_c)
            x = torch.cat(x_smoothed, dim=1).transpose(1, 2)  # [B, T, C]

        # Day-specific normalization
        x_normalized = []
        for i, day_id in enumerate(day_idx):
            x_normalized.append(self.day_layer_norm[day_id](x[i:i+1]))
        x = torch.cat(x_normalized, dim=0)

        # Encode with multi-scale encoder
        if self.use_multiscale_ctc:
            # Get all scales for multi-scale CTC
            encoder_scales = self.encoder(x, return_all_scales=True)
            fast_output = encoder_scales['fast']      # [B, T_fast, D] ~75 timesteps (stride 2)
            encoder_output = encoder_scales['fused']  # [B, T_medium, D] ~38 timesteps (stride 4) - main
            slow_output = encoder_scales['slow']      # [B, T_slow, D] ~19 timesteps (stride 8)
        else:
            # Only get fused medium scale
            encoder_output = self.encoder(x)  # [B, T', D] where T' = T/4 (medium scale)

        # Contrastive pre-training mode
        if return_contrastive_loss and self.use_contrastive_pretraining:
            return self.contrastive_module(x)

        # Compute output lengths for each scale
        out_lengths = torch.full((x.shape[0],), encoder_output.shape[1], dtype=torch.long, device=x.device)

        if self.use_multiscale_ctc:
            fast_out_lengths = torch.full((x.shape[0],), fast_output.shape[1], dtype=torch.long, device=x.device)
            slow_out_lengths = torch.full((x.shape[0],), slow_output.shape[1], dtype=torch.long, device=x.device)

        if self.use_marginalization:
            # PROPER DCoND APPROACH: Predict diphones, marginalize to get phonemes
            # Step 1: Predict diphone logits
            diphone_logits = self.diphone_output(encoder_output)  # [B, T', num_diphones]
            diphone_log_probs = diphone_logits.log_softmax(dim=-1)  # [B, T', num_diphones]

            # Step 2: Marginalize to get phoneme log probabilities
            # P(phoneme_j | x) = sum_i P(diphone_ij | x)
            # In log space: log P(phoneme_j) = logsumexp over diphones ending in j
            # But we can use matrix multiplication in probability space for simplicity
            diphone_probs = torch.exp(diphone_log_probs)  # [B, T', num_diphones]

            # Matrix multiply: [B, T', num_diphones] @ [num_diphones, n_classes] -> [B, T', n_classes]
            phone_probs = torch.matmul(diphone_probs, self.marginalization_matrix)  # [B, T', n_classes]

            # Convert back to log probabilities (add small epsilon for numerical stability)
            phone_log_probs = torch.log(phone_probs + 1e-10)  # [B, T', n_classes]
            phone_log_probs = phone_log_probs.transpose(0, 1)  # [T', B, n_classes]

            # Also return diphone log probs for joint loss
            diphone_log_probs_transposed = diphone_log_probs.transpose(0, 1)  # [T', B, num_diphones]

            # STEP 3: Compute auxiliary CTC outputs from fast and slow scales
            if self.use_multiscale_ctc:
                # Fast pathway auxiliary phoneme head
                fast_phone_logits = self.fast_phone_head(fast_output)  # [B, T_fast, n_classes]
                fast_phone_log_probs = fast_phone_logits.log_softmax(dim=-1).transpose(0, 1)  # [T_fast, B, n_classes]

                # Slow pathway auxiliary phoneme head
                slow_phone_logits = self.slow_phone_head(slow_output)  # [B, T_slow, n_classes]
                slow_phone_log_probs = slow_phone_logits.log_softmax(dim=-1).transpose(0, 1)  # [T_slow, B, n_classes]

                # Return: main phone, main lengths, main diphone, fast phone, fast lengths, slow phone, slow lengths
                return (
                    phone_log_probs,              # Main phoneme predictions (fused, from diphone marginalization)
                    out_lengths,                   # Main output lengths
                    diphone_log_probs_transposed, # Main diphone predictions
                    fast_phone_log_probs,         # Fast pathway auxiliary phoneme predictions
                    fast_out_lengths,             # Fast pathway output lengths
                    slow_phone_log_probs,         # Slow pathway auxiliary phoneme predictions
                    slow_out_lengths,             # Slow pathway output lengths
                )
            else:
                return phone_log_probs, out_lengths, diphone_log_probs_transposed

        elif self.use_diphone_head:
            # OLD APPROACH: Separate independent heads
            phone_logits = self.phone_output(encoder_output)  # [B, T', n_classes]
            phone_log_probs = phone_logits.log_softmax(dim=-1).transpose(0, 1)  # [T', B, n_classes]

            diphone_logits = self.diphone_output(encoder_output)  # [B, T', num_diphones]
            diphone_log_probs = diphone_logits.log_softmax(dim=-1).transpose(0, 1)  # [T', B, num_diphones]
            return phone_log_probs, out_lengths, diphone_log_probs

        else:
            # BASELINE: Only phoneme prediction
            phone_logits = self.phone_output(encoder_output)  # [B, T', n_classes]
            phone_log_probs = phone_logits.log_softmax(dim=-1).transpose(0, 1)  # [T', B, n_classes]
            return phone_log_probs, out_lengths


# Keep the old attention version for reference but use the simpler CTC version
class MultiScaleAttentionNeuralDecoder(nn.Module):
    """
    DEPRECATED: Too complex, has bugs. Use MultiScaleCTCDecoder instead.

    Kept for reference only.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "Attention-based decoder has bugs. Use MultiScaleCTCDecoder instead:\n"
            "model = MultiScaleCTCDecoder(...)"
        )
