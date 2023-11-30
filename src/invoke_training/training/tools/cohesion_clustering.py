from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


@dataclass
class Cluster:
    label: int
    image_paths: list[str]
    mean_dist_from_center: float


def _embed_images(
    image_paths: list[Path], clip_image_encoder: CLIPVisionModelWithProjection, device: torch.device, dtype: torch.dtype
) -> np.ndarray:
    clip_image_processor = CLIPImageProcessor()

    # Calculate CLIP image embeddings of all images.
    image_embeds = []
    for image_path in image_paths:
        image_pil = [Image.open(image_path)]
        image_clip = clip_image_processor(images=image_pil, return_tensors="pt").pixel_values
        clip_image_embeds = clip_image_encoder(image_clip.to(device, dtype=dtype)).image_embeds
        image_embeds.append(clip_image_embeds.detach().clone().cpu().numpy())

    # Flatten the embedding for each image.
    image_embeds = np.array(image_embeds)
    image_embeds = image_embeds.reshape((len(image_embeds), -1))
    return image_embeds


def _calculate_cluster_cohesion(image_embeds: np.ndarray, cluster_centers: np.ndarray, cluster_labels: np.ndarray):
    cluster_cohesions: list[float] = []
    for label in range(len(cluster_centers)):
        cluster_embeds = image_embeds[cluster_labels == label]
        center_dists = np.linalg.norm(cluster_embeds - cluster_centers[label], axis=-1)
        cluster_cohesions.append(center_dists.mean())
    return cluster_cohesions


def cluster_images(
    image_paths: list[Path],
    clip_image_encoder: CLIPVisionModelWithProjection,
    device: torch.device,
    dtype: torch.dtype,
    target_cluster_size: int,
) -> list[Cluster]:
    num_clusters = len(image_paths) // target_cluster_size
    assert num_clusters > 1

    image_embeds = _embed_images(image_paths, clip_image_encoder, device, dtype)

    # Run k-means++
    # TODO(ryand): The paper is not totally clear about what distance metric is used for k-means and how the centroids
    # are calculated. There is one sentence that says that cosine similarity is used, but it's not clear if this means
    # spherical k-means or some other implementation. Euclidean distance is used later to determine cohesion.
    k_means = KMeans(n_clusters=num_clusters, init="k-means++").fit(image_embeds)

    cluster_cohesions = _calculate_cluster_cohesion(image_embeds, k_means.cluster_centers_, k_means.labels_)

    # Construct Cluster objects.
    clusters = [
        Cluster(label=i, image_paths=[], mean_dist_from_center=cluster_cohesions[i]) for i in range(num_clusters)
    ]
    for i, image_path in enumerate(image_paths):
        clusters[k_means.labels_[i]].image_paths.append(image_path)

    return clusters


def choose_best_cluster(clusters: list[Cluster], min_cluster_size: int) -> Cluster:
    # Filter small clusters
    clusters = [c for c in clusters if len(c.image_paths) >= min_cluster_size]
    # Sort to find lowest mean cohesion distance.
    clusters = sorted(clusters, key=lambda c: c.mean_dist_from_center)

    return clusters[0]
