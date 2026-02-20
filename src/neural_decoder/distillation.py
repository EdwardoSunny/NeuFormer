"""
D – Knowledge Distillation: Teacher → Student Reranker
======================================================
D1: Collect training data from the teacher (LLM-scored N-best).
D2: Train a small student discriminative reranker.
D3: Deploy the student for fast inference.

Two student architectures are provided:

1. **StudentReranker** (MLP, default): A lightweight calibration/reranking
   head that takes candidate-level uncertainty + neural score features
   (9-dim) and produces a scalar score. Cheap to evaluate (sub-ms), runs
   on CPU in real time. Recovers partial teacher gains. Best framed as
   a *calibration head* that combines posterior uncertainty with neural
   acoustics — not as a full language model replacement.

2. **TransformerStudentReranker** (optional): A small transformer encoder
   (~5-20M params) that reads tokenised candidate text and produces a
   score. Stronger than the MLP but requires GPU; still much smaller
   than the teacher LLM and suitable for on-device deployment with
   quantisation.

Both are trained with a pairwise ranking loss (hinge-style) so that
they mimic the teacher's preference ordering.

Usage
-----
    # D1 – collect data
    from neural_decoder.distillation import DistillationDataCollector
    collector = DistillationDataCollector(llm_scorer)
    dataset = collector.collect(utterance_candidates)

    # D2 – train student
    from neural_decoder.distillation import StudentReranker, train_student
    student = StudentReranker(feature_dim=...)
    train_student(student, dataset)

    # D3 – deploy
    best = student.rerank(candidates, features)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


# ======================================================================
# D1 – Data structures & collection
# ======================================================================


@dataclass
class CandidateFeatures:
    """Feature vector for one candidate hypothesis."""

    neural_score: float = 0.0
    lm_score: float = 0.0
    n_words: int = 0
    mean_entropy: float = 0.0
    min_margin: float = 0.0
    blank_ratio: float = 0.0
    n_uncertain_slots: int = 0
    n_locked_slots: int = 0
    edit_distance_to_best: int = 0

    def to_numpy(self) -> np.ndarray:
        return np.array(
            [
                self.neural_score,
                self.lm_score,
                self.n_words,
                self.mean_entropy,
                self.min_margin,
                self.blank_ratio,
                self.n_uncertain_slots,
                self.n_locked_slots,
                self.edit_distance_to_best,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def feature_dim() -> int:
        return 9


@dataclass
class DistillationSample:
    """One training sample: a set of candidates for one utterance."""

    candidates: List[List[str]]  # word sequences
    features: List[CandidateFeatures]  # feature vectors
    teacher_scores: List[float]  # teacher combined scores
    best_idx: int = 0  # index of teacher-preferred candidate


@dataclass
class DistillationDataset:
    """Collection of distillation samples."""

    samples: List[DistillationSample] = field(default_factory=list)

    def __len__(self):
        return len(self.samples)

    def get_pair_dataset(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Convert to pairwise training data: (best_features, other_features, margin).

        Returns list of (feat_best, feat_other, score_margin) tuples.
        """
        pairs = []
        for sample in self.samples:
            if len(sample.candidates) < 2:
                continue
            best_feat = sample.features[sample.best_idx].to_numpy()
            best_score = sample.teacher_scores[sample.best_idx]
            for i, (feat, score) in enumerate(
                zip(sample.features, sample.teacher_scores)
            ):
                if i == sample.best_idx:
                    continue
                pairs.append((best_feat, feat.to_numpy(), best_score - score))
        return pairs


class DistillationDataCollector:
    """
    D1 – Collects training data for distillation.

    Given an LLM scorer and per-utterance candidates, computes teacher
    scores and packages everything into a DistillationDataset.
    """

    def __init__(
        self,
        llm_score_fn: Callable[[str], float],
        lambda_neural: float = 1.0,
        lambda_lm: float = 0.5,
    ):
        self.llm_score_fn = llm_score_fn
        self.lambda_neural = lambda_neural
        self.lambda_lm = lambda_lm

    def collect_sample(
        self,
        candidates: List[List[str]],
        neural_scores: List[float],
        uncertainty_features: Optional[List[Dict]] = None,
    ) -> DistillationSample:
        """
        Score candidates with the teacher and build a DistillationSample.

        Parameters
        ----------
        candidates : list of word lists
        neural_scores : parallel list of neural model scores
        uncertainty_features : optional list of dicts with keys like
            'mean_entropy', 'min_margin', 'blank_ratio',
            'n_uncertain_slots', 'n_locked_slots'
        """
        feats_list: List[CandidateFeatures] = []
        teacher_scores: List[float] = []

        for i, (words, ns) in enumerate(zip(candidates, neural_scores)):
            sentence = " ".join(words)
            lm_score = self.llm_score_fn(sentence)

            uf = uncertainty_features[i] if uncertainty_features else {}

            feat = CandidateFeatures(
                neural_score=ns,
                lm_score=lm_score,
                n_words=len(words),
                mean_entropy=uf.get("mean_entropy", 0.0),
                min_margin=uf.get("min_margin", 0.0),
                blank_ratio=uf.get("blank_ratio", 0.0),
                n_uncertain_slots=uf.get("n_uncertain_slots", 0),
                n_locked_slots=uf.get("n_locked_slots", 0),
                edit_distance_to_best=uf.get("edit_distance_to_best", 0),
            )
            feats_list.append(feat)

            combined = self.lambda_neural * ns + self.lambda_lm * lm_score
            teacher_scores.append(combined)

        best_idx = int(np.argmax(teacher_scores))

        return DistillationSample(
            candidates=candidates,
            features=feats_list,
            teacher_scores=teacher_scores,
            best_idx=best_idx,
        )


# ======================================================================
# D2 – Student Reranker
# ======================================================================


class StudentReranker(nn.Module):
    """
    Calibration/reranking head (MLP).

    Takes candidate-level features (uncertainty metrics + neural scores)
    and produces scalar scores for reranking.  This is intentionally
    lightweight: it combines posterior uncertainty information with
    acoustic model scores to approximate the teacher LLM's preference
    ordering.

    It is *not* a language model — it operates on pre-computed feature
    vectors, making it sub-millisecond at inference.  Expected to
    recover partial teacher gains (typically 30-60% of the WER
    improvement from LLM rescoring) at negligible computational cost.
    """

    def __init__(self, feature_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, feature_dim]
        returns: [B, 1] scalar scores
        """
        return self.net(x)

    @torch.no_grad()
    def rerank(
        self,
        candidates: List[List[str]],
        features: List[CandidateFeatures],
    ) -> Tuple[List[str], float]:
        """
        D3 – fast inference reranking.

        Returns the best candidate and its score.
        """
        self.eval()
        feat_np = np.stack([f.to_numpy() for f in features])
        feat_t = torch.from_numpy(feat_np).float()
        if next(self.parameters()).is_cuda:
            feat_t = feat_t.cuda()
        scores = self.forward(feat_t).squeeze(-1)
        best_idx = scores.argmax().item()
        return candidates[best_idx], scores[best_idx].item()


def train_student(
    student: StudentReranker,
    dataset: DistillationDataset,
    n_epochs: int = 50,
    lr: float = 1e-3,
    margin: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
) -> List[float]:
    """
    D2 – Train the student reranker with pairwise ranking loss.

    Loss = max(0, margin - (score_best - score_other))

    Returns list of per-epoch average losses.
    """
    pairs = dataset.get_pair_dataset()
    if not pairs:
        if verbose:
            print("No training pairs available.")
        return []

    student = student.to(device)
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    epoch_losses: List[float] = []

    for epoch in range(n_epochs):
        np.random.shuffle(pairs)
        total_loss = 0.0
        n_pairs = 0

        for best_feat, other_feat, score_margin in pairs:
            best_t = torch.from_numpy(best_feat).unsqueeze(0).to(device)
            other_t = torch.from_numpy(other_feat).unsqueeze(0).to(device)

            s_best = student(best_t).squeeze()
            s_other = student(other_t).squeeze()

            loss = torch.clamp(margin - (s_best - s_other), min=0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_pairs += 1

        avg_loss = total_loss / max(n_pairs, 1)
        epoch_losses.append(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}, loss: {avg_loss:.4f}")

    student.eval()
    return epoch_losses


# ======================================================================
# D2b – Optional Transformer Student Reranker
# ======================================================================


class TransformerStudentReranker(nn.Module):
    """
    Lightweight transformer-based reranker (~5-20M params).

    Reads tokenised candidate text and produces a scalar score.
    Stronger than the MLP calibration head, but requires GPU for
    reasonable latency.  Still much smaller than the teacher LLM
    and suitable for on-device deployment with quantisation.

    Architecture: small transformer encoder → mean-pool → linear → score.

    Parameters
    ----------
    vocab_size : int
        Tokeniser vocabulary size.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    max_len : int
        Maximum sequence length.
    dropout : float
        Dropout rate.
    feature_dim : int
        Size of auxiliary feature vector (appended to pooled representation).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
        feature_dim: int = 9,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model + feature_dim, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : [B, L] int tensor
        attention_mask : [B, L] bool tensor (True = attend)
        features : [B, feature_dim] float tensor (auxiliary features)

        Returns
        -------
        scores : [B, 1] float tensor
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.embedding(token_ids) + self.pos_embedding(positions)

        # Create causal-free mask for encoder
        if attention_mask is not None:
            # TransformerEncoder expects src_key_padding_mask: True = ignore
            pad_mask = ~attention_mask
        else:
            pad_mask = None

        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # Mean pool over non-padded positions
        if attention_mask is not None:
            mask_float = attention_mask.float().unsqueeze(-1)  # [B, L, 1]
            pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)  # [B, d_model]

        if features is not None:
            pooled = torch.cat([pooled, features], dim=-1)

        return self.head(pooled)

    @torch.no_grad()
    def rerank(
        self,
        candidates: List[List[str]],
        token_ids_list: List[torch.Tensor],
        features: Optional[List[CandidateFeatures]] = None,
    ) -> Tuple[List[str], float]:
        """
        Rerank candidates using the transformer student.

        Parameters
        ----------
        candidates : list of word lists
        token_ids_list : list of 1D token ID tensors (one per candidate)
        features : optional list of CandidateFeatures

        Returns
        -------
        (best_word_list, best_score)
        """
        self.eval()
        if not candidates:
            return [], 0.0

        # Pad token_ids to same length
        max_len = max(t.shape[0] for t in token_ids_list)
        B = len(token_ids_list)
        padded = torch.zeros(B, max_len, dtype=torch.long)
        mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, t in enumerate(token_ids_list):
            padded[i, : t.shape[0]] = t
            mask[i, : t.shape[0]] = True

        device = next(self.parameters()).device
        padded = padded.to(device)
        mask = mask.to(device)

        feat_tensor = None
        if features is not None:
            feat_np = np.stack([f.to_numpy() for f in features])
            feat_tensor = torch.from_numpy(feat_np).float().to(device)

        scores = self.forward(padded, mask, feat_tensor).squeeze(-1)
        best_idx = scores.argmax().item()
        return candidates[best_idx], scores[best_idx].item()
