from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluation import compute_retrieval_metrics


class ResidualFeatureAdapter(nn.Module):
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
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.fc2(self.dropout(self.activation(self.fc1(x))))
        return F.normalize(x + self.beta * delta, dim=-1)


class TopKRetrievalReranker(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 512,
        score_hidden_dim: int | None = None,
        dropout: float = 0.1,
        beta: float = 0.1,
        share_adapters: bool = False,
        scorer_type: str = "cosine",
        use_top1_head: bool = False,
        top1_head_hidden_dim: int | None = None,
        top1_score_coef: float = 0.25,
    ) -> None:
        super().__init__()
        if scorer_type not in {"cosine", "mlp_pairwise", "contextual_transformer", "listwise_transformer"}:
            raise ValueError(f"Unknown scorer_type: {scorer_type}")
        self.scorer_type = scorer_type
        self.use_top1_head = bool(use_top1_head)
        self.top1_score_coef = float(top1_score_coef)
        self.query_adapter = ResidualFeatureAdapter(
            feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            beta=beta,
        )
        if share_adapters:
            self.target_adapter = self.query_adapter
        else:
            self.target_adapter = ResidualFeatureAdapter(
                feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                beta=beta,
            )
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))
        if scorer_type == "mlp_pairwise":
            scorer_hidden_dim = int(score_hidden_dim or hidden_dim)
            self.score_mlp = nn.Sequential(
                nn.Linear(feature_dim * 4 + 2, scorer_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(scorer_hidden_dim, scorer_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(scorer_hidden_dim, 1),
            )
            final_layer = self.score_mlp[-1]
            assert isinstance(final_layer, nn.Linear)
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        elif scorer_type == "contextual_transformer":
            scorer_hidden_dim = int(score_hidden_dim or hidden_dim)
            self.context_input = nn.Linear(feature_dim * 4 + 2, scorer_hidden_dim)
            self.context_norm = nn.LayerNorm(scorer_hidden_dim)
            self.shortlist_position_embedding = nn.Embedding(32, scorer_hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=scorer_hidden_dim,
                nhead=8,
                dim_feedforward=scorer_hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.context_head = nn.Linear(scorer_hidden_dim, 1)
            nn.init.zeros_(self.context_head.weight)
            nn.init.zeros_(self.context_head.bias)
        elif scorer_type == "listwise_transformer":
            scorer_hidden_dim = int(score_hidden_dim or hidden_dim)
            self.listwise_query_input = nn.Linear(feature_dim, scorer_hidden_dim)
            self.listwise_candidate_input = nn.Linear(feature_dim * 4 + 2, scorer_hidden_dim)
            self.listwise_norm = nn.LayerNorm(scorer_hidden_dim)
            self.listwise_position_embedding = nn.Embedding(33, scorer_hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=scorer_hidden_dim,
                nhead=8,
                dim_feedforward=scorer_hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.listwise_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.listwise_head = nn.Sequential(
                nn.Linear(scorer_hidden_dim * 4, scorer_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(scorer_hidden_dim, 1),
            )
            final_layer = self.listwise_head[-1]
            assert isinstance(final_layer, nn.Linear)
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        if self.use_top1_head:
            top1_hidden_dim = int(top1_head_hidden_dim or score_hidden_dim or hidden_dim)
            self.top1_logit_scale = nn.Parameter(torch.tensor(0.0))
            self.top1_head = nn.Sequential(
                nn.Linear(feature_dim * 4 + 2, top1_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(top1_hidden_dim, top1_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(top1_hidden_dim, 1),
            )
            final_layer = self.top1_head[-1]
            assert isinstance(final_layer, nn.Linear)
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def encode_query(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        return self.query_adapter(query_embeddings)

    def encode_target(self, target_embeddings: torch.Tensor) -> torch.Tensor:
        return self.target_adapter(target_embeddings)

    def _build_pairwise_features(
        self,
        query_expanded: torch.Tensor,
        target: torch.Tensor,
        base_shortlist_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cosine_score = (query_expanded * target).sum(dim=-1, keepdim=True)
        features = torch.cat(
            [
                query_expanded,
                target,
                query_expanded * target,
                (query_expanded - target).abs(),
                cosine_score,
                base_shortlist_scores.unsqueeze(-1),
            ],
            dim=-1,
        )
        return features, cosine_score

    def score_all(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        query = self.encode_query(query_embeddings)
        target = self.encode_target(candidate_embeddings)
        return scale * query @ target.T

    def score_shortlist(
        self,
        query_embeddings: torch.Tensor,
        shortlist_candidate_embeddings: torch.Tensor,
        *,
        base_shortlist_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.score_shortlist_details(
            query_embeddings,
            shortlist_candidate_embeddings,
            base_shortlist_scores=base_shortlist_scores,
        )["combined_scores"]

    def score_shortlist_details(
        self,
        query_embeddings: torch.Tensor,
        shortlist_candidate_embeddings: torch.Tensor,
        *,
        base_shortlist_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if shortlist_candidate_embeddings.ndim != 3:
            raise ValueError("Expected shortlist_candidate_embeddings to have shape [batch, topk, dim].")
        batch_size, topk, feature_dim = shortlist_candidate_embeddings.shape
        query = self.encode_query(query_embeddings)
        target = self.encode_target(shortlist_candidate_embeddings.reshape(batch_size * topk, feature_dim))
        target = target.reshape(batch_size, topk, feature_dim)
        query_expanded = query.unsqueeze(1).expand(-1, topk, -1)

        require_base_scores = self.scorer_type != "cosine" or self.use_top1_head
        if require_base_scores and base_shortlist_scores is None:
            raise ValueError("base_shortlist_scores are required for this scorer configuration.")

        pairwise_features: torch.Tensor | None = None
        cosine_score: torch.Tensor | None = None
        if base_shortlist_scores is not None:
            pairwise_features, cosine_score = self._build_pairwise_features(
                query_expanded,
                target,
                base_shortlist_scores,
            )

        ranking_scores: torch.Tensor
        if self.scorer_type == "cosine":
            scale = self.logit_scale.exp().clamp(max=100.0)
            if cosine_score is None:
                cosine_score = (query_expanded * target).sum(dim=-1, keepdim=True)
            ranking_scores = scale * cosine_score.squeeze(-1)
        elif self.scorer_type == "mlp_pairwise":
            if pairwise_features is None:
                raise ValueError("pairwise_features must be available for non-cosine scorer types.")
            ranking_scores = self.score_mlp(pairwise_features).squeeze(-1)
        elif self.scorer_type == "contextual_transformer":
            if pairwise_features is None:
                raise ValueError("pairwise_features must be available for non-cosine scorer types.")
            position_ids = torch.arange(topk, device=pairwise_features.device)
            if topk > int(self.shortlist_position_embedding.num_embeddings):
                raise ValueError(
                    f"shortlist_topk={topk} exceeds the contextual scorer capacity "
                    f"({self.shortlist_position_embedding.num_embeddings})."
                )
            context_tokens = self.context_input(pairwise_features)
            context_tokens = self.context_norm(context_tokens)
            context_tokens = context_tokens + self.shortlist_position_embedding(position_ids).unsqueeze(0)
            context_tokens = self.context_encoder(context_tokens)
            ranking_scores = self.context_head(context_tokens).squeeze(-1)
        else:
            if pairwise_features is None:
                raise ValueError("pairwise_features must be available for non-cosine scorer types.")
            if topk + 1 > int(self.listwise_position_embedding.num_embeddings):
                raise ValueError(
                    f"shortlist_topk={topk} exceeds the listwise scorer capacity "
                    f"({self.listwise_position_embedding.num_embeddings - 1})."
                )
            query_token = self.listwise_query_input(query).unsqueeze(1)
            candidate_tokens = self.listwise_candidate_input(pairwise_features)
            tokens = torch.cat([query_token, candidate_tokens], dim=1)
            position_ids = torch.arange(topk + 1, device=tokens.device)
            tokens = self.listwise_norm(tokens)
            tokens = tokens + self.listwise_position_embedding(position_ids).unsqueeze(0)
            tokens = self.listwise_encoder(tokens)
            query_context = tokens[:, :1].expand(-1, topk, -1)
            candidate_context = tokens[:, 1:]
            pair_context = torch.cat(
                [
                    candidate_context,
                    query_context,
                    candidate_context * query_context,
                    (candidate_context - query_context).abs(),
                ],
                dim=-1,
            )
            ranking_scores = self.listwise_head(pair_context).squeeze(-1)

        top1_logits = None
        combined_scores = ranking_scores
        if self.use_top1_head:
            if pairwise_features is None:
                raise ValueError("pairwise_features must be available when use_top1_head=True.")
            top1_scale = self.top1_logit_scale.exp().clamp(max=100.0)
            top1_logits = top1_scale * self.top1_head(pairwise_features).squeeze(-1)
            combined_scores = combined_scores + float(self.top1_score_coef) * top1_logits

        return {
            "ranking_scores": ranking_scores,
            "top1_logits": top1_logits,
            "combined_scores": combined_scores,
        }


def build_shortlist_indices(
    base_logits: torch.Tensor,
    *,
    candidate_image_ids: list[str],
    query_image_ids: Iterable[str] | None = None,
    shortlist_topk: int = 5,
    ensure_positive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if shortlist_topk < 1:
        raise ValueError("shortlist_topk must be >= 1.")
    resolved_topk = min(int(shortlist_topk), base_logits.shape[1])
    shortlist_indices = base_logits.topk(k=resolved_topk, dim=1).indices.clone()

    if not ensure_positive:
        return shortlist_indices, None
    if query_image_ids is None:
        raise ValueError("query_image_ids are required when ensure_positive=True.")

    target_map = {image_id: idx for idx, image_id in enumerate(candidate_image_ids)}
    target_positions = torch.zeros(base_logits.shape[0], dtype=torch.long)
    for row_idx, image_id in enumerate(query_image_ids):
        target_idx = target_map[image_id]
        row = shortlist_indices[row_idx]
        matches = (row == target_idx).nonzero(as_tuple=False)
        if len(matches) > 0:
            target_positions[row_idx] = int(matches[0, 0].item())
            continue
        row[-1] = target_idx
        target_positions[row_idx] = resolved_topk - 1
    return shortlist_indices, target_positions


def shortlist_cross_entropy_loss(
    shortlist_scores: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(shortlist_scores, target_positions)


def shortlist_margin_loss(
    shortlist_scores: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    positive_scores = shortlist_scores.gather(1, target_positions.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(shortlist_scores, dtype=torch.bool)
    mask.scatter_(1, target_positions.unsqueeze(1), False)
    hardest_negative = shortlist_scores.masked_fill(~mask, float("-inf")).max(dim=1).values
    return F.relu(float(margin) + hardest_negative - positive_scores).mean()


def shortlist_pairwise_logistic_loss(
    shortlist_scores: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    rank_weight_power: float = 1.0,
) -> torch.Tensor:
    batch_size, shortlist_topk = shortlist_scores.shape
    positive_scores = shortlist_scores.gather(1, target_positions.unsqueeze(1))
    pairwise_margins = positive_scores - shortlist_scores

    mask = torch.ones_like(shortlist_scores, dtype=torch.bool)
    mask.scatter_(1, target_positions.unsqueeze(1), False)
    pairwise_losses = F.softplus(-pairwise_margins)

    base_rank = torch.arange(shortlist_topk, device=shortlist_scores.device, dtype=shortlist_scores.dtype)
    negative_weights = 1.0 / (base_rank + 1.0).pow(float(rank_weight_power))
    negative_weights = negative_weights.unsqueeze(0).expand(batch_size, -1)
    negative_weights = negative_weights.masked_fill(~mask, 0.0)
    negative_weights = negative_weights / negative_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    weighted_losses = (pairwise_losses * negative_weights).sum(dim=1)
    return weighted_losses.mean()


def shortlist_focal_cross_entropy_loss(
    shortlist_scores: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    gamma: float = 2.0,
) -> torch.Tensor:
    log_probs = F.log_softmax(shortlist_scores, dim=-1)
    target_log_probs = log_probs.gather(1, target_positions.unsqueeze(1)).squeeze(1)
    target_probs = target_log_probs.exp()
    modulation = (1.0 - target_probs).clamp_min(1e-6).pow(float(gamma))
    return -(modulation * target_log_probs).mean()


def shortlist_hard_negative_logistic_loss(
    shortlist_scores: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    positive_scores = shortlist_scores.gather(1, target_positions.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(shortlist_scores, dtype=torch.bool)
    mask.scatter_(1, target_positions.unsqueeze(1), False)
    hardest_negative = shortlist_scores.masked_fill(~mask, float("-inf")).max(dim=1).values
    return F.softplus(hardest_negative - positive_scores).mean()


def reranker_loss(
    shortlist_scores: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    auxiliary_top1_scores: torch.Tensor | None = None,
    ce_loss_coef: float = 1.0,
    margin_loss_coef: float = 0.0,
    pairwise_loss_coef: float = 0.0,
    focal_loss_coef: float = 0.0,
    hard_negative_logistic_loss_coef: float = 0.0,
    top1_ce_loss_coef: float = 0.0,
    top1_focal_loss_coef: float = 0.0,
    top1_hard_negative_loss_coef: float = 0.0,
    focal_gamma: float = 2.0,
    pairwise_rank_weight_power: float = 1.0,
    margin: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce_loss = shortlist_cross_entropy_loss(shortlist_scores, target_positions)
    margin_loss = shortlist_margin_loss(shortlist_scores, target_positions, margin=margin)
    pairwise_loss = shortlist_pairwise_logistic_loss(
        shortlist_scores,
        target_positions,
        rank_weight_power=pairwise_rank_weight_power,
    )
    focal_loss = shortlist_focal_cross_entropy_loss(shortlist_scores, target_positions, gamma=focal_gamma)
    hard_negative_logistic_loss = shortlist_hard_negative_logistic_loss(shortlist_scores, target_positions)
    total = ce_loss * float(ce_loss_coef)
    total = total + margin_loss * float(margin_loss_coef)
    total = total + pairwise_loss * float(pairwise_loss_coef)
    total = total + focal_loss * float(focal_loss_coef)
    total = total + hard_negative_logistic_loss * float(hard_negative_logistic_loss_coef)
    top1_ce_loss = shortlist_scores.new_zeros(())
    top1_focal_loss = shortlist_scores.new_zeros(())
    top1_hard_negative_loss = shortlist_scores.new_zeros(())
    if auxiliary_top1_scores is not None:
        top1_ce_loss = shortlist_cross_entropy_loss(auxiliary_top1_scores, target_positions)
        top1_focal_loss = shortlist_focal_cross_entropy_loss(
            auxiliary_top1_scores,
            target_positions,
            gamma=focal_gamma,
        )
        top1_hard_negative_loss = shortlist_hard_negative_logistic_loss(
            auxiliary_top1_scores,
            target_positions,
        )
        total = total + top1_ce_loss * float(top1_ce_loss_coef)
        total = total + top1_focal_loss * float(top1_focal_loss_coef)
        total = total + top1_hard_negative_loss * float(top1_hard_negative_loss_coef)
    return total, {
        "ce_loss": float(ce_loss.item()),
        "margin_loss": float(margin_loss.item()),
        "pairwise_loss": float(pairwise_loss.item()),
        "focal_loss": float(focal_loss.item()),
        "hard_negative_logistic_loss": float(hard_negative_logistic_loss.item()),
        "top1_ce_loss": float(top1_ce_loss.item()),
        "top1_focal_loss": float(top1_focal_loss.item()),
        "top1_hard_negative_loss": float(top1_hard_negative_loss.item()),
        "total_loss": float(total.item()),
    }


def reorder_logits_within_shortlist(
    base_logits: torch.Tensor,
    shortlist_indices: torch.Tensor,
    order_scores: torch.Tensor,
) -> torch.Tensor:
    if shortlist_indices.shape != order_scores.shape:
        raise ValueError("shortlist_indices and order_scores must have the same shape.")
    base_shortlist_scores = base_logits.gather(1, shortlist_indices)
    sorted_base_scores = base_shortlist_scores.sort(dim=1, descending=True).values
    rerank_order = order_scores.argsort(dim=1, descending=True)
    reassigned_scores = torch.zeros_like(sorted_base_scores).scatter(1, rerank_order, sorted_base_scores)
    reranked_logits = base_logits.clone()
    reranked_logits.scatter_(1, shortlist_indices, reassigned_scores)
    return reranked_logits


def select_rerank_weight(
    *,
    base_logits: torch.Tensor,
    rerank_scores: torch.Tensor,
    shortlist_indices: torch.Tensor,
    ordered_image_ids: list[str],
    candidate_image_ids: list[str],
    weight_grid: Iterable[float] | None = None,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    search_grid = list(weight_grid or [0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    shortlist_base_scores = base_logits.gather(1, shortlist_indices)
    best_weight = float(search_grid[0])
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float] | None = None
    history: list[dict[str, float]] = []

    for weight in search_grid:
        order_scores = shortlist_base_scores + float(weight) * rerank_scores
        logits = reorder_logits_within_shortlist(base_logits, shortlist_indices, order_scores)
        metrics = compute_retrieval_metrics(
            logits,
            ordered_image_ids=ordered_image_ids,
            candidate_image_ids=candidate_image_ids,
        )
        payload = {"weight": float(weight), **metrics}
        history.append(payload)
        score = (
            float(metrics["top1_acc"]),
            float(metrics["top5_acc"]),
            -abs(float(weight) - 1.0),
        )
        if best_score is None or score > best_score:
            best_weight = float(weight)
            best_metrics = metrics
            best_score = score

    assert best_metrics is not None
    return best_weight, best_metrics, history
