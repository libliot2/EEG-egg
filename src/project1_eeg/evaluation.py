from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy as sp
import torch
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torch import nn
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


def _forward_in_batches(
    inputs: torch.Tensor,
    forward_fn,
    *,
    batch_size: int = 64,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start : start + batch_size]
        outputs.append(torch.as_tensor(forward_fn(chunk)).detach())
    return torch.cat(outputs, dim=0)


def _feature_in_batches(
    inputs: torch.Tensor,
    model,
    *,
    feature_layer: str,
    batch_size: int = 64,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start : start + batch_size]
        outputs.append(torch.as_tensor(model(chunk)[feature_layer]).detach())
    return torch.cat(outputs, dim=0)


def compute_retrieval_metrics(
    logits: torch.Tensor,
    *,
    ordered_image_ids: list[str] | None = None,
    candidate_image_ids: list[str] | None = None,
) -> dict[str, float]:
    if logits.ndim != 2:
        raise ValueError("Expected a rank-2 similarity matrix.")

    if ordered_image_ids is None:
        if logits.shape[0] != logits.shape[1]:
            raise ValueError("Square logits are required when image ids are not provided.")
        targets = torch.arange(logits.shape[0], device=logits.device)
    else:
        if candidate_image_ids is None:
            candidate_image_ids = ordered_image_ids
        target_map = {image_id: idx for idx, image_id in enumerate(candidate_image_ids)}
        targets = torch.tensor([target_map[image_id] for image_id in ordered_image_ids], device=logits.device)

    top1_pred = logits.argmax(dim=1)
    top1_acc = (top1_pred == targets).float().mean().item()

    top5_idx = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
    top5_acc = (top5_idx == targets[:, None]).any(dim=1).float().mean().item()

    return {"top1_acc": top1_acc, "top5_acc": top5_acc}


def rank_candidate_ids(logits: torch.Tensor, candidate_image_ids: list[str]) -> list[list[str]]:
    ranking = logits.argsort(dim=1, descending=True).cpu().tolist()
    return [[candidate_image_ids[idx] for idx in row] for row in ranking]


@torch.no_grad()
def two_way_identification(
    all_brain_recons: torch.Tensor,
    all_images: torch.Tensor,
    model: nn.Module | Any,
    preprocess: transforms.Compose,
    feature_layer: str | None = None,
    return_avg: bool = True,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 64,
) -> float | tuple[np.ndarray, int]:
    recon_batch = torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device)
    real_batch = torch.stack([preprocess(image) for image in all_images], dim=0).to(device)

    if feature_layer is None:
        preds = _forward_in_batches(recon_batch, model, batch_size=batch_size)
        reals = _forward_in_batches(real_batch, model, batch_size=batch_size)
    else:
        preds = _feature_in_batches(recon_batch, model, feature_layer=feature_layer, batch_size=batch_size)
        reals = _feature_in_batches(real_batch, model, feature_layer=feature_layer, batch_size=batch_size)

    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()

    corr = np.corrcoef(reals, preds)
    corr = corr[: len(all_images), len(all_images) :]
    congruent = np.diag(corr)
    success = corr < congruent
    success_count = np.sum(success, axis=0)

    if return_avg:
        return float(np.mean(success_count) / (len(all_images) - 1))
    return success_count, len(all_images) - 1


def pixcorr(all_images: torch.Tensor, all_brain_recons: torch.Tensor) -> float:
    preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    flat_images = preprocess(all_images).reshape(len(all_images), -1).cpu()
    flat_recons = preprocess(all_brain_recons).reshape(len(all_brain_recons), -1).cpu()

    scores = []
    for idx in tqdm(range(min(len(flat_images), len(flat_recons))), desc="pixcorr", leave=False):
        scores.append(np.corrcoef(flat_images[idx], flat_recons[idx])[0][1])
    return float(np.mean(scores))


def ssim_score(all_images: torch.Tensor, all_brain_recons: torch.Tensor) -> float:
    preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    image_gray = rgb2gray(preprocess(all_images).permute((0, 2, 3, 1)).cpu().numpy())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0, 2, 3, 1)).cpu().numpy())

    scores = []
    for image, recon in tqdm(zip(image_gray, recon_gray), total=len(all_images), desc="ssim", leave=False):
        scores.append(
            structural_similarity(
                recon,
                image,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0,
            )
        )
    return float(np.mean(scores))


def alexnet_identification(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    from torchvision.models import AlexNet_Weights, alexnet

    weights = AlexNet_Weights.IMAGENET1K_V1
    model = create_feature_extractor(alexnet(weights=weights), return_nodes=["features.4", "features.11"]).to(device)
    model.eval().requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    alex2 = two_way_identification(all_brain_recons, all_images, model, preprocess, "features.4", device=device)
    alex5 = two_way_identification(all_brain_recons, all_images, model, preprocess, "features.11", device=device)
    return float(alex2), float(alex5)


def inception_identification(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
) -> float:
    weights = Inception_V3_Weights.DEFAULT
    model = create_feature_extractor(inception_v3(weights=weights), return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    score = two_way_identification(all_brain_recons, all_images, model, preprocess, "avgpool", device=device)
    return float(score)


def clip_identification(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
) -> float:
    import clip

    clip_model, _ = clip.load("ViT-L/14", device=device)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    score = two_way_identification(all_brain_recons, all_images, clip_model.encode_image, preprocess, None, device=device)
    return float(score)


def effnet_correlation(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

    weights = EfficientNet_B1_Weights.DEFAULT
    model = create_feature_extractor(efficientnet_b1(weights=weights), return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gt = _feature_in_batches(
        preprocess(all_images),
        model,
        feature_layer="avgpool",
        batch_size=batch_size,
    ).reshape(len(all_images), -1).cpu().numpy()
    fake = _feature_in_batches(
        preprocess(all_brain_recons),
        model,
        feature_layer="avgpool",
        batch_size=batch_size,
    ).reshape(len(all_brain_recons), -1).cpu().numpy()
    scores = [sp.spatial.distance.correlation(gt[idx], fake[idx]) for idx in range(len(gt))]
    return float(np.mean(scores))


def swav_correlation(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    try:
        model = torch.hub.load("facebookresearch/swav:main", "resnet50")
    except Exception as exc:  # pragma: no cover - external download path
        warnings.warn(f"SWAV metric failed to load: {exc}")
        return float("nan")

    model = create_feature_extractor(model, return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    gt = _feature_in_batches(
        preprocess(all_images),
        model,
        feature_layer="avgpool",
        batch_size=batch_size,
    ).reshape(len(all_images), -1).cpu().numpy()
    fake = _feature_in_batches(
        preprocess(all_brain_recons),
        model,
        feature_layer="avgpool",
        batch_size=batch_size,
    ).reshape(len(all_brain_recons), -1).cpu().numpy()
    scores = [sp.spatial.distance.correlation(gt[idx], fake[idx]) for idx in range(len(gt))]
    return float(np.mean(scores))


def eval_images(
    *,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device | str = torch.device("cpu"),
) -> dict[str, float]:
    device = torch.device(device)
    real_images = real_images.to(device).float()
    fake_images = fake_images.to(device).float()

    alex2, alex5 = alexnet_identification(real_images, fake_images, device=device)
    return {
        "eval_pixcorr": float(pixcorr(real_images, fake_images)),
        "eval_ssim": float(ssim_score(real_images, fake_images)),
        "eval_alex2": float(alex2),
        "eval_alex5": float(alex5),
        "eval_inception": float(inception_identification(real_images, fake_images, device=device)),
        "eval_clip": float(clip_identification(real_images, fake_images, device=device)),
        "eval_effnet": float(effnet_correlation(real_images, fake_images, device=device)),
        "eval_swav": float(swav_correlation(real_images, fake_images, device=device)),
    }
