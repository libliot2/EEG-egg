from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeResidualModel(nn.Module):
    def __init__(
        self,
        *,
        eeg_encoder: nn.Module | None = None,
        embedding_dim: int = 768,
        prototype_channels: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if eeg_encoder is None:
            raise ValueError("PrototypeResidualModel requires an EEG embedder module.")
        self.eeg_encoder = eeg_encoder
        self.prototype_adapter = nn.Sequential(
            nn.Conv2d(prototype_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.scale = nn.Linear(embedding_dim, hidden_dim)
        self.shift = nn.Linear(embedding_dim, hidden_dim)
        self.residual_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, prototype_channels, kernel_size=3, padding=1),
        )

    def forward(self, eeg: torch.Tensor, prototype_latent: torch.Tensor) -> torch.Tensor:
        eeg_embedding = self.eeg_encoder(eeg)
        adapted = self.prototype_adapter(prototype_latent)
        scale = self.scale(eeg_embedding).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(eeg_embedding).unsqueeze(-1).unsqueeze(-1)
        fused = adapted * (1.0 + scale) + shift
        residual = self.residual_head(fused)
        return prototype_latent + residual


class EEGEmbeddingRegressor(nn.Module):
    def __init__(
        self,
        *,
        eeg_encoder: nn.Module | None = None,
        backbone_dim: int = 768,
        target_dim: int = 1280,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if eeg_encoder is None:
            raise ValueError("EEGEmbeddingRegressor requires an EEG encoder module.")
        self.eeg_encoder = eeg_encoder
        self.head = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.head(self.eeg_encoder(eeg))

    def encoder_parameters(self):
        return self.eeg_encoder.parameters()

    def head_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("eeg_encoder."):
                yield parameter


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        query = query + attended
        query = query + self.ffn(self.output_norm(query))
        return query


@dataclass
class GatedRetrievalResidualOutputs:
    direct_embedding: torch.Tensor
    final_embedding: torch.Tensor
    retrieval_embedding: torch.Tensor
    residual_embedding: torch.Tensor
    gate: torch.Tensor
    confidence: torch.Tensor | None = None
    predicted_text_embedding: torch.Tensor | None = None


class GatedRetrievalResidualRegressor(nn.Module):
    def __init__(
        self,
        *,
        eeg_encoder: nn.Module | None = None,
        backbone_dim: int = 768,
        target_dim: int = 1280,
        retrieval_dim: int = 1792,
        text_dim: int | None = None,
        hidden_dim: int = 1024,
        attention_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if eeg_encoder is None:
            raise ValueError("GatedRetrievalResidualRegressor requires an EEG encoder module.")

        self.eeg_encoder = eeg_encoder
        self.text_dim = text_dim

        self.direct_head = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, target_dim),
        )
        self.retrieval_head = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, retrieval_dim),
        )
        self.text_head = None
        if text_dim is not None:
            self.text_head = nn.Sequential(
                nn.LayerNorm(backbone_dim),
                nn.Linear(backbone_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, text_dim),
            )

        self.query_projection = nn.Linear(backbone_dim, hidden_dim)
        self.image_context_projection = nn.Linear(target_dim, hidden_dim)
        self.text_context_projection = None if text_dim is None else nn.Linear(text_dim, hidden_dim)
        self.context_block = CrossAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, target_dim),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.final_norm = nn.LayerNorm(target_dim)

    def encoder_parameters(self):
        return self.eeg_encoder.parameters()

    def head_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("eeg_encoder."):
                yield parameter

    def encode_backbone(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.eeg_encoder(eeg)

    def project_shared(
        self,
        shared: torch.Tensor,
        *,
        retrieved_embeddings: torch.Tensor | None = None,
        retrieved_text_embeddings: torch.Tensor | None = None,
        retrieval_confidence: torch.Tensor | None = None,
    ) -> GatedRetrievalResidualOutputs:
        direct_embedding = self.direct_head(shared)
        retrieval_embedding = F.normalize(self.retrieval_head(shared), dim=-1)

        predicted_text_embedding = None
        if self.text_head is not None:
            predicted_text_embedding = F.normalize(self.text_head(shared), dim=-1)

        residual_embedding = torch.zeros_like(direct_embedding)
        gate = torch.zeros((shared.shape[0], 1), device=shared.device, dtype=direct_embedding.dtype)
        confidence = None

        if retrieved_embeddings is not None:
            context_tokens = [self.image_context_projection(retrieved_embeddings)]
            if (
                retrieved_text_embeddings is not None
                and self.text_context_projection is not None
            ):
                context_tokens.append(self.text_context_projection(retrieved_text_embeddings))
            context = torch.cat(context_tokens, dim=1)
            query = self.query_projection(shared).unsqueeze(1)
            fused = self.context_block(query, context).squeeze(1)
            residual_embedding = self.residual_head(fused)
            gate = torch.sigmoid(self.gate_head(fused))
            if retrieval_confidence is not None:
                confidence = retrieval_confidence.to(device=shared.device, dtype=gate.dtype).view(-1, 1)
                gate = gate * confidence

        final_embedding = self.final_norm(direct_embedding + gate * residual_embedding)
        return GatedRetrievalResidualOutputs(
            direct_embedding=direct_embedding,
            final_embedding=final_embedding,
            retrieval_embedding=retrieval_embedding,
            residual_embedding=residual_embedding,
            gate=gate,
            confidence=confidence,
            predicted_text_embedding=predicted_text_embedding,
        )

    def forward(
        self,
        eeg: torch.Tensor,
        *,
        retrieved_embeddings: torch.Tensor | None = None,
        retrieved_text_embeddings: torch.Tensor | None = None,
        retrieval_confidence: torch.Tensor | None = None,
    ) -> GatedRetrievalResidualOutputs:
        shared = self.encode_backbone(eeg)
        return self.project_shared(
            shared,
            retrieved_embeddings=retrieved_embeddings,
            retrieved_text_embeddings=retrieved_text_embeddings,
            retrieval_confidence=retrieval_confidence,
        )
