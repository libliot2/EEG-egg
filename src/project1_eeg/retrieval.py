from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGPerturbation(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, in_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


class TargetFeatureAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int,
        dropout: float = 0.1,
        beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.net(x)
        adapted = (1.0 - self.beta) * x + self.beta * (x + delta)
        return F.normalize(adapted, dim=-1)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class EEGEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 63,
        hidden_dim: int = 256,
        embedding_dim: int = 768,
        channel_dropout: float = 0.1,
        time_mask_ratio: float = 0.1,
        use_eeg_perturbation: bool = False,
    ) -> None:
        super().__init__()
        self.channel_dropout = channel_dropout
        self.time_mask_ratio = time_mask_ratio
        self.eeg_perturbation = EEGPerturbation(in_channels) if use_eeg_perturbation else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(hidden_dim, dilation=1),
            ResidualTemporalBlock(hidden_dim, dilation=2),
            ResidualTemporalBlock(hidden_dim, dilation=4),
            ResidualTemporalBlock(hidden_dim, dilation=8),
        )
        self.attention = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.channel_dropout > 0.0:
            mask = torch.rand(x.shape[:2], device=x.device) > self.channel_dropout
            x = x * mask.unsqueeze(-1)
        if self.training and self.time_mask_ratio > 0.0:
            time_steps = x.shape[-1]
            mask_width = max(1, int(time_steps * self.time_mask_ratio))
            starts = torch.randint(0, max(1, time_steps - mask_width + 1), (x.shape[0],), device=x.device)
            for batch_idx, start in enumerate(starts.tolist()):
                x[batch_idx, :, start : start + mask_width] = 0.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._augment(x)
        x = self.eeg_perturbation(x)
        features = self.blocks(self.stem(x))
        attn = torch.softmax(self.attention(features), dim=-1)
        pooled = torch.sum(features * attn, dim=-1)
        mean = features.mean(dim=-1)
        embedding = self.projection(torch.cat([pooled, mean], dim=-1))
        return F.normalize(embedding, dim=-1)


class ATMEncoder(nn.Module):
    def __init__(
        self,
        *,
        variant: str = "small",
        in_channels: int = 63,
        hidden_dim: int = 256,
        embedding_dim: int = 768,
        channel_dropout: float = 0.1,
        time_mask_ratio: float = 0.1,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        dropout: float = 0.1,
        use_eeg_perturbation: bool = False,
    ) -> None:
        super().__init__()
        if variant not in {"small", "base", "large"}:
            raise ValueError(f"Unknown ATM encoder variant: {variant}")
        self.channel_dropout = channel_dropout
        self.time_mask_ratio = time_mask_ratio
        self.output_dim = embedding_dim
        self.variant = variant
        self.eeg_perturbation = EEGPerturbation(in_channels) if use_eeg_perturbation else nn.Identity()

        temporal_dilations = {
            "small": (1, 2, 4),
            "base": (1, 2, 4, 8),
            "large": (1, 2, 4, 8, 16),
        }[variant]
        frequency_dilations = {
            "small": (1, 2),
            "base": (1, 2, 4),
            "large": (1, 2, 4, 8),
        }[variant]
        frequency_hidden_dim = max(hidden_dim // 2, 64)

        self.temporal_stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.temporal_blocks = nn.Sequential(
            *(ResidualTemporalBlock(hidden_dim, dilation=dilation) for dilation in temporal_dilations)
        )
        self.temporal_attention = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.transformer_norm = nn.LayerNorm(hidden_dim)

        self.frequency_branch = nn.Sequential(
            nn.Conv1d(in_channels, frequency_hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(frequency_hidden_dim),
            nn.GELU(),
            *(ResidualTemporalBlock(frequency_hidden_dim, dilation=dilation) for dilation in frequency_dilations),
        )

        fusion_dim = hidden_dim * 2 + frequency_hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.channel_dropout > 0.0:
            mask = torch.rand(x.shape[:2], device=x.device) > self.channel_dropout
            x = x * mask.unsqueeze(-1)
        if self.training and self.time_mask_ratio > 0.0:
            time_steps = x.shape[-1]
            mask_width = max(1, int(time_steps * self.time_mask_ratio))
            starts = torch.randint(0, max(1, time_steps - mask_width + 1), (x.shape[0],), device=x.device)
            for batch_idx, start in enumerate(starts.tolist()):
                x[batch_idx, :, start : start + mask_width] = 0.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._augment(x)
        x = self.eeg_perturbation(x)

        temporal = self.temporal_blocks(self.temporal_stem(x))
        temporal_attn = torch.softmax(self.temporal_attention(temporal), dim=-1)
        temporal_pooled = torch.sum(temporal * temporal_attn, dim=-1)

        tokens = F.avg_pool1d(temporal, kernel_size=4, stride=4).transpose(1, 2)
        transformed = self.transformer_norm(self.transformer(tokens)).mean(dim=1)

        spectrum = torch.log1p(torch.fft.rfft(x, dim=-1).abs())
        frequency = self.frequency_branch(spectrum).mean(dim=-1)

        fused = torch.cat([temporal_pooled, transformed, frequency], dim=-1)
        return self.projection(fused)


class ATMSmallEncoder(ATMEncoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="small", **kwargs)


class ATMBaseEncoder(ATMEncoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="base", **kwargs)


class ATMLargeEncoder(ATMEncoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="large", **kwargs)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = max(input_dim, min(output_dim, 1024))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class BackboneHeadEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(eeg))


@dataclass
class RetrievalOutputs:
    semantic: torch.Tensor | None = None
    perceptual: torch.Tensor | None = None
    legacy: torch.Tensor | None = None

    def default(self) -> torch.Tensor:
        if self.semantic is not None:
            return self.semantic
        if self.perceptual is not None:
            return self.perceptual
        if self.legacy is not None:
            return self.legacy
        raise RuntimeError("RetrievalOutputs is empty.")


class RetrievalModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 63,
        hidden_dim: int = 256,
        embedding_dim: int = 768,
        channel_dropout: float = 0.1,
        time_mask_ratio: float = 0.1,
        encoder_type: str = "legacy_cnn",
        semantic_dim: int | None = None,
        perceptual_dim: int | None = None,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        dropout: float = 0.1,
        use_eeg_perturbation: bool = False,
        use_semantic_target_adapter: bool = False,
        use_perceptual_target_adapter: bool = False,
        target_adapter_hidden_dim: int = 1024,
        target_adapter_dropout: float = 0.1,
        target_adapter_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.semantic_dim = semantic_dim
        self.perceptual_dim = perceptual_dim
        self.target_adapter_beta = float(target_adapter_beta)
        self.legacy_mode = (
            encoder_type == "legacy_cnn"
            and perceptual_dim is None
            and (semantic_dim is None or semantic_dim == embedding_dim)
        )

        if self.legacy_mode:
            self.encoder = EEGEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                channel_dropout=channel_dropout,
                time_mask_ratio=time_mask_ratio,
                use_eeg_perturbation=use_eeg_perturbation,
            )
            self.logit_scale = nn.Parameter(torch.tensor(2.6593))
            return

        if encoder_type == "legacy_cnn":
            raise ValueError("Dual-head training requires an ATM encoder variant, not encoder_type='legacy_cnn'.")
        if semantic_dim is None and perceptual_dim is None:
            raise ValueError("At least one retrieval head must be enabled.")

        encoder_cls = {
            "atm_small": ATMSmallEncoder,
            "atm_base": ATMBaseEncoder,
            "atm_large": ATMLargeEncoder,
        }.get(encoder_type)
        if encoder_cls is None:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.encoder = encoder_cls(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            channel_dropout=channel_dropout,
            time_mask_ratio=time_mask_ratio,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
            use_eeg_perturbation=use_eeg_perturbation,
        )
        if semantic_dim is not None:
            self.semantic_head = ProjectionHead(embedding_dim, semantic_dim, dropout=dropout)
            self.semantic_logit_scale = nn.Parameter(torch.tensor(2.6593))
            if use_semantic_target_adapter:
                self.semantic_target_adapter = TargetFeatureAdapter(
                    semantic_dim,
                    hidden_dim=target_adapter_hidden_dim,
                    dropout=target_adapter_dropout,
                    beta=target_adapter_beta,
                )
        if perceptual_dim is not None:
            self.perceptual_head = ProjectionHead(embedding_dim, perceptual_dim, dropout=dropout)
            self.perceptual_logit_scale = nn.Parameter(torch.tensor(2.6593))
            if use_perceptual_target_adapter:
                self.perceptual_target_adapter = TargetFeatureAdapter(
                    perceptual_dim,
                    hidden_dim=target_adapter_hidden_dim,
                    dropout=target_adapter_dropout,
                    beta=target_adapter_beta,
                )

    def has_head(self, head: str) -> bool:
        if head == "legacy":
            return self.legacy_mode
        return hasattr(self, f"{head}_head")

    def primary_head(self) -> str:
        if self.has_head("semantic"):
            return "semantic"
        if self.has_head("perceptual"):
            return "perceptual"
        return "legacy"

    def output_dim(self, head: str | None = None) -> int:
        head = head or self.primary_head()
        if head == "legacy":
            return self.embedding_dim
        dim = getattr(self, f"{head}_dim", None)
        if dim is None:
            raise KeyError(f"Unknown retrieval head: {head}")
        return int(dim)

    def encode_all(self, eeg: torch.Tensor) -> RetrievalOutputs:
        if self.legacy_mode:
            return RetrievalOutputs(legacy=self.encoder(eeg))

        shared = self.encoder(eeg)
        outputs = RetrievalOutputs()
        if self.has_head("semantic"):
            outputs.semantic = self.semantic_head(shared)
        if self.has_head("perceptual"):
            outputs.perceptual = self.perceptual_head(shared)
        return outputs

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.encode_all(eeg).default()

    def build_primary_embedder(self) -> nn.Module:
        if self.legacy_mode:
            return copy.deepcopy(self.encoder)
        head_name = self.primary_head()
        head = getattr(self, f"{head_name}_head")
        return BackboneHeadEncoder(copy.deepcopy(self.encoder), copy.deepcopy(head))

    def similarity(
        self,
        eeg_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        *,
        head: str | None = None,
    ) -> torch.Tensor:
        head = head or self.primary_head()
        image_embeddings = self.adapt_target_embeddings(image_embeddings, head=head)
        if head == "legacy":
            scale = self.logit_scale.exp().clamp(max=100.0)
        else:
            scale = getattr(self, f"{head}_logit_scale").exp().clamp(max=100.0)
        return scale * eeg_embeddings @ image_embeddings.T

    def adapt_target_embeddings(
        self,
        image_embeddings: torch.Tensor,
        *,
        head: str | None = None,
    ) -> torch.Tensor:
        head = head or self.primary_head()
        if head == "legacy":
            return image_embeddings
        adapter = getattr(self, f"{head}_target_adapter", None)
        if adapter is None:
            return image_embeddings
        return adapter(image_embeddings)

    def target_adapter_regularization(
        self,
        image_embeddings: torch.Tensor,
        *,
        head: str | None = None,
    ) -> torch.Tensor | None:
        head = head or self.primary_head()
        if head == "legacy":
            return None
        adapter = getattr(self, f"{head}_target_adapter", None)
        if adapter is None:
            return None
        adapted = adapter(image_embeddings)
        baseline = F.normalize(image_embeddings, dim=-1)
        return F.mse_loss(adapted, baseline)


def retrieval_loss(
    model: RetrievalModel,
    eeg_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    *,
    head: str | None = None,
) -> torch.Tensor:
    logits = model.similarity(eeg_embeddings, image_embeddings, head=head)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) * 0.5


def soft_clip_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def build_soft_targets(
    eeg_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    eeg_similarity = F.normalize(eeg_embeddings, dim=-1) @ F.normalize(eeg_embeddings, dim=-1).T
    image_similarity = F.normalize(image_embeddings, dim=-1) @ F.normalize(image_embeddings, dim=-1).T
    blended_similarity = 0.5 * (eeg_similarity + image_similarity)
    return torch.softmax(blended_similarity * beta, dim=-1)


def off_diagonal_log_probs(logits: torch.Tensor) -> torch.Tensor:
    mask = ~torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    masked = logits.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(masked, dim=-1)
    return log_probs.masked_fill(~mask, 0.0)


def relation_alignment_loss(
    logits: torch.Tensor,
    eeg_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    targets = build_soft_targets(eeg_embeddings, image_embeddings, beta=beta)
    mask = ~torch.eye(targets.shape[0], device=targets.device, dtype=torch.bool)
    masked_targets = targets.masked_fill(~mask, 0.0)
    masked_targets = masked_targets / masked_targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(masked_targets * off_diagonal_log_probs(logits)).sum(dim=-1).mean()


def neuroclip_retrieval_loss(
    model: RetrievalModel,
    eeg_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    *,
    head: str | None = None,
    soft_target_beta: float = 10.0,
    clip_loss_coef: float = 1.0,
    soft_loss_coef: float = 0.3,
    relation_loss_coef: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model.similarity(eeg_embeddings, image_embeddings, head=head)
    targets = torch.arange(logits.shape[0], device=logits.device)
    clip_loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) * 0.5

    soft_targets = build_soft_targets(eeg_embeddings, image_embeddings, beta=soft_target_beta)
    soft_loss = (soft_clip_loss(logits, soft_targets) + soft_clip_loss(logits.T, soft_targets.T)) * 0.5

    relation_loss = (
        relation_alignment_loss(logits, eeg_embeddings, image_embeddings, beta=soft_target_beta)
        + relation_alignment_loss(logits.T, image_embeddings, eeg_embeddings, beta=soft_target_beta)
    ) * 0.5

    total = clip_loss * clip_loss_coef
    total = total + soft_loss * soft_loss_coef
    total = total + relation_loss * relation_loss_coef
    return total, {
        "clip_loss": float(clip_loss.item()),
        "soft_loss": float(soft_loss.item()),
        "relation_loss": float(relation_loss.item()),
    }


def weighted_retrieval_loss(
    model: RetrievalModel,
    outputs: RetrievalOutputs,
    *,
    semantic_targets: torch.Tensor | None = None,
    perceptual_targets: torch.Tensor | None = None,
    semantic_weight: float = 1.0,
    perceptual_weight: float = 0.7,
    retrieval_loss_type: str = "clip",
    soft_target_beta: float = 10.0,
    clip_loss_coef: float = 1.0,
    soft_loss_coef: float = 0.3,
    relation_loss_coef: float = 0.05,
    target_adapter_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    total_loss: torch.Tensor | None = None
    metrics: dict[str, float] = {}

    def compute_loss(
        eeg_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        *,
        head: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if retrieval_loss_type == "clip":
            loss = retrieval_loss(model, eeg_embeddings, image_embeddings, head=head)
            return loss, {}
        if retrieval_loss_type == "neuroclip":
            return neuroclip_retrieval_loss(
                model,
                eeg_embeddings,
                image_embeddings,
                head=head,
                soft_target_beta=soft_target_beta,
                clip_loss_coef=clip_loss_coef,
                soft_loss_coef=soft_loss_coef,
                relation_loss_coef=relation_loss_coef,
            )
        raise ValueError(f"Unknown retrieval_loss_type: {retrieval_loss_type}")

    if outputs.legacy is not None:
        if semantic_targets is None:
            raise ValueError("semantic_targets are required for legacy retrieval.")
        loss, extra_metrics = compute_loss(outputs.legacy, semantic_targets, head="legacy")
        total_loss = loss
        metrics["legacy_loss"] = float(loss.item())
        for key, value in extra_metrics.items():
            metrics[f"legacy_{key}"] = value
    else:
        if outputs.semantic is not None and semantic_targets is not None and semantic_weight > 0.0:
            semantic_loss, extra_metrics = compute_loss(outputs.semantic, semantic_targets, head="semantic")
            total_loss = semantic_loss * semantic_weight if total_loss is None else total_loss + semantic_loss * semantic_weight
            metrics["semantic_loss"] = float(semantic_loss.item())
            for key, value in extra_metrics.items():
                metrics[f"semantic_{key}"] = value
            semantic_adapter_reg = model.target_adapter_regularization(semantic_targets, head="semantic")
            if semantic_adapter_reg is not None and target_adapter_loss_weight > 0.0:
                total_loss = total_loss + semantic_adapter_reg * target_adapter_loss_weight
                metrics["semantic_target_adapter_reg"] = float(semantic_adapter_reg.item())
        if outputs.perceptual is not None and perceptual_targets is not None and perceptual_weight > 0.0:
            perceptual_loss, extra_metrics = compute_loss(outputs.perceptual, perceptual_targets, head="perceptual")
            total_loss = (
                perceptual_loss * perceptual_weight
                if total_loss is None
                else total_loss + perceptual_loss * perceptual_weight
            )
            metrics["perceptual_loss"] = float(perceptual_loss.item())
            for key, value in extra_metrics.items():
                metrics[f"perceptual_{key}"] = value
            perceptual_adapter_reg = model.target_adapter_regularization(perceptual_targets, head="perceptual")
            if perceptual_adapter_reg is not None and target_adapter_loss_weight > 0.0:
                total_loss = total_loss + perceptual_adapter_reg * target_adapter_loss_weight
                metrics["perceptual_target_adapter_reg"] = float(perceptual_adapter_reg.item())

    if total_loss is None:
        raise ValueError("No retrieval loss terms were enabled.")

    metrics["total_loss"] = float(total_loss.item())
    return total_loss, metrics


def build_retrieval_model_from_config(config: dict) -> RetrievalModel:
    return RetrievalModel(
        in_channels=int(config["in_channels"]),
        hidden_dim=int(config["hidden_dim"]),
        embedding_dim=int(config["embedding_dim"]),
        channel_dropout=float(config.get("channel_dropout", 0.1)),
        time_mask_ratio=float(config.get("time_mask_ratio", 0.1)),
        encoder_type=str(config.get("encoder_type", "legacy_cnn")),
        semantic_dim=config.get("semantic_dim"),
        perceptual_dim=config.get("perceptual_dim"),
        transformer_layers=int(config.get("transformer_layers", 2)),
        transformer_heads=int(config.get("transformer_heads", 8)),
        dropout=float(config.get("dropout", 0.1)),
        use_eeg_perturbation=bool(config.get("use_eeg_perturbation", False)),
        use_semantic_target_adapter=bool(config.get("use_semantic_target_adapter", False)),
        use_perceptual_target_adapter=bool(config.get("use_perceptual_target_adapter", False)),
        target_adapter_hidden_dim=int(config.get("target_adapter_hidden_dim", 1024)),
        target_adapter_dropout=float(config.get("target_adapter_dropout", 0.1)),
        target_adapter_beta=float(config.get("target_adapter_beta", 0.1)),
    )
