#!/usr/bin/env python3
"""
Pose2ID Paper Implementation - Feature Centralization for Person Re-ID Clustering
Based on: "From Poses to Identity: Training-Free Person Re-Identification via Feature Centralization"
CVPR 2025 - https://github.com/yuanc3/Pose2ID
"""

from src.index.hnsw_index import HNSWIndex
from ultralytics import YOLO
from src.reid.osnet_reid import OSNetReID
from config.global_config import apply_preset, get_config
import torch
import numpy as np
import cv2
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN
import sys
import os

# Add project paths
sys.path.append('.')
sys.path.append('src')


# --------------------------------------------------------------------------------------
# Pose2ID Paper Implementation - NFC (Neighbor Feature Centralization)
# --------------------------------------------------------------------------------------


def pairwise_distance(query_features, gallery_features):
    """Compute pairwise distance matrix (from Pose2ID paper)"""
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def NFC(feat: torch.tensor, k1=2, k2=2):
    """
    Neighbor Feature Centralization (NFC) - Core technique from Pose2ID paper

    Args:
        feat: Feature tensor [N, D]
        k1: Number of nearest neighbors to consider
        k2: Number of mutual neighbors required

    Returns:
        Centralized features
    """
    device = feat.device
    feat = feat.clone()

    # Compute pairwise distances
    dist = pairwise_distance(feat, feat)

    # Mask diagonal (self-distances)
    eye = torch.eye(dist.size(0)).to(device)
    dist[eye == 1] = 1000  # Large value to exclude self

    # Find k1 nearest neighbors
    val, rank = dist.topk(k1, largest=False)

    # Find mutual top-k neighbors
    mutual_topk_list = []
    for i in range(rank.size(0)):
        mutual_list = []
        for j in rank[i]:
            if i in rank[j][:k2]:
                mutual_list.append(j.item())
        mutual_topk_list.append(mutual_list)

    # Feature centralization: add neighbor features
    feat_copy = feat.clone()
    for i in range(rank.size(0)):
        if mutual_topk_list[i]:  # Only if mutual neighbors exist
            feat[i] += feat_copy[mutual_topk_list[i]].sum(dim=0)

    return feat

# --------------------------------------------------------------------------------------
# ID² Metric (Identity Density) - From Pose2ID paper
# --------------------------------------------------------------------------------------


def ID2(features, pids):
    """
    ID² Metric - Identity Density from Pose2ID paper
    Quantitative metric to replace t-SNE visualization

    Args:
        features: Feature matrix [N, D]
        pids: Person IDs [N]

    Returns:
        Identity density scores
    """
    unique_pids = np.unique(pids)
    densities = []

    for pid in unique_pids:
        mask = pids == pid
        if np.sum(mask) < 2:
            densities.append(0.0)
            continue

        pid_features = features[mask]

        # Compute intra-class distances (same identity)
        intra_distances = []
        for i in range(len(pid_features)):
            for j in range(i+1, len(pid_features)):
                dist = np.linalg.norm(pid_features[i] - pid_features[j])
                intra_distances.append(dist)

        # Compute inter-class distances (different identities)
        inter_distances = []
        other_features = features[~mask]
        for feat in pid_features:
            for other_feat in other_features:
                dist = np.linalg.norm(feat - other_feat)
                inter_distances.append(dist)

        # Identity density = inter_distance / intra_distance
        if intra_distances:
            avg_intra = np.mean(intra_distances)
            avg_inter = np.mean(inter_distances) if inter_distances else 1.0
            density = avg_inter / (avg_intra + 1e-8)
        else:
            density = 0.0

        densities.append(density)

    return np.array(densities)

# --------------------------------------------------------------------------------------
# Pose2ID Clustering Implementation
# --------------------------------------------------------------------------------------


def pose2id_clustering(features, k1=4, k2=4, cluster_method='kmeans', n_clusters=5):
    """
    Pose2ID clustering pipeline following the paper methodology

    Args:
        features: Input features [N, D]
        k1, k2: NFC parameters
        cluster_method: 'kmeans' or 'dbscan'
        n_clusters: Number of expected clusters

    Returns:
        labels, centralized_features, metrics
    """
    print(f"🔬 Pose2ID Clustering Pipeline")
    print(f"   - Input features: {features.shape}")
    print(f"   - NFC params: k1={k1}, k2={k2}")
    print(f"   - Clustering method: {cluster_method}")

    # Step 1: Apply NFC (Neighbor Feature Centralization)
    print("📐 Applying NFC (Neighbor Feature Centralization)...")
    features_tensor = torch.tensor(features).float()
    features_nfc = NFC(features_tensor, k1=k1, k2=k2)
    features_nfc = features_nfc.numpy()

    # Step 2: L2 Normalization (as mentioned in paper)
    print("🔄 L2 Normalization...")
    features_nfc = features_nfc / \
        (np.linalg.norm(features_nfc, axis=1, keepdims=True) + 1e-8)

    # Step 3: Clustering
    print(f"🎯 Clustering with {cluster_method}...")
    if cluster_method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(features_nfc)
    elif cluster_method == 'dbscan':
        # Adaptive eps based on feature space
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=4)
        neighbors_fit = neighbors.fit(features_nfc)
        distances, indices = neighbors_fit.kneighbors(features_nfc)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        eps = np.percentile(distances, 70)  # Adaptive eps

        clusterer = DBSCAN(eps=eps, min_samples=max(3, len(features_nfc)//50))
        labels = clusterer.fit_predict(features_nfc)

        # Handle noise points
        noise_mask = labels == -1
        if np.any(noise_mask):
            print(f"   - Noise points detected: {np.sum(noise_mask)}")
            # Assign noise points to nearest cluster
            for i in np.where(noise_mask)[0]:
                distances_to_clusters = []
                for cluster_id in set(labels) - {-1}:
                    cluster_mask = labels == cluster_id
                    if np.any(cluster_mask):
                        cluster_center = features_nfc[cluster_mask].mean(
                            axis=0)
                        dist = np.linalg.norm(features_nfc[i] - cluster_center)
                        distances_to_clusters.append((dist, cluster_id))

                if distances_to_clusters:
                    _, nearest_cluster = min(distances_to_clusters)
                    labels[i] = nearest_cluster

    # Step 4: Compute metrics
    print("📊 Computing metrics...")
    metrics = {}

    # Silhouette Score
    if len(set(labels)) > 1:
        metrics['silhouette_score'] = silhouette_score(features_nfc, labels)
    else:
        metrics['silhouette_score'] = 0.0

    # Cluster statistics
    unique_labels = set(labels)
    metrics['n_clusters'] = len(unique_labels)
    metrics['cluster_sizes'] = {int(label): int(
        np.sum(labels == label)) for label in unique_labels}

    # Feature centralization improvement
    # Compare original vs NFC features
    original_silhouette = silhouette_score(
        features, labels) if len(set(labels)) > 1 else 0.0
    nfc_silhouette = metrics['silhouette_score']
    metrics['nfc_improvement'] = nfc_silhouette - original_silhouette

    print(f"✅ Clustering completed:")
    print(f"   - Clusters found: {metrics['n_clusters']}")
    print(f"   - Silhouette score: {metrics['silhouette_score']:.3f}")
    print(f"   - NFC improvement: {metrics['nfc_improvement']:.3f}")

    return labels, features_nfc, metrics

# --------------------------------------------------------------------------------------
# Video Processing
# --------------------------------------------------------------------------------------


def extract_person_crops(video_path: Path, confidence=0.9, frame_skip=2, max_frames=None):
    """Extract person crops from video"""
    yolo_model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video {video_path}')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    crops, stats = [], defaultdict(int)
    stats.update({'confidence_threshold': confidence})

    pbar = tqdm(total=total_frames//frame_skip, desc='📹 Extracting frames')
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % frame_skip == 0:
            stats['total_frames_processed'] += 1
            results = yolo_model(frame, verbose=False)

            for res in results:
                if res.boxes is None:
                    continue

                # Person class == 0
                for box in res.boxes[res.boxes.cls == 0]:
                    stats['total_detections'] += 1
                    conf = box.conf.item()
                    if conf < confidence:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Add padding
                    pad = 0.1
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, x1 - int(w * pad))
                    y1 = max(0, y1 - int(h * pad))
                    x2 = min(width, x2 + int(w * pad))
                    y2 = min(height, y2 + int(h * pad))

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
                        continue

                    crops.append({
                        'crop': crop,
                        'frame': frame_no,
                        'conf': conf,
                        'ts': frame_no / fps
                    })
                    stats['valid_detections'] += 1

            pbar.update(1)
            if max_frames and frame_no >= max_frames:
                break

        frame_no += 1

    pbar.close()
    cap.release()

    return crops, stats


def extract_features(crops, extractor):
    """Extract features using ReID model"""
    features, metadata = [], []

    for idx, item in enumerate(tqdm(crops, desc='🔍 Extracting features')):
        emb = extractor.extract_features(item['crop'])
        if emb is not None and len(emb) > 0:
            features.append(emb[0])
            metadata.append({
                'idx': idx,
                'frame': item['frame'],
                'conf': item['conf'],
                'ts': item['ts']
            })

    return np.array(features), metadata

# --------------------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------------------


def create_pose2id_visualization(features_original, features_nfc, labels, metadata, output_dir):
    """Create comprehensive visualization following Pose2ID paper style"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)

    # Original features
    original_2d = pca.fit_transform(features_original)

    # NFC features
    nfc_2d = pca.fit_transform(features_nfc)

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    unique_labels = sorted(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    # Plot original features
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax1.scatter(original_2d[mask, 0], original_2d[mask, 1],
                    c=[colors[i]], label=f'Person {label}', alpha=0.7, s=30)

    ax1.set_title('Original Features (Before NFC)',
                  fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot NFC features
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax2.scatter(nfc_2d[mask, 0], nfc_2d[mask, 1],
                    c=[colors[i]], label=f'Person {label}', alpha=0.7, s=30)

    ax2.set_title('NFC Features (After Centralization)',
                  fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = output_dir / 'pose2id_feature_centralization.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    return viz_path

# --------------------------------------------------------------------------------------
# Main Pipeline
# --------------------------------------------------------------------------------------


def main(video_path: str, output_dir: str, expected_persons: int = 5,
         k1: int = 4, k2: int = 4, cluster_method: str = 'kmeans',
         frame_skip: int = 2, max_frames: int = None):
    """
    Main Pose2ID clustering pipeline

    Args:
        video_path: Path to input video
        output_dir: Output directory
        expected_persons: Expected number of persons
        k1, k2: NFC parameters
        cluster_method: 'kmeans' or 'dbscan'
        frame_skip: Process every N frames
        max_frames: Maximum frames to process
    """

    print("🚀 Pose2ID Paper Implementation - Feature Centralization Clustering")
    print("=" * 70)

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply configuration
    apply_preset('three_person_dataset_90_confidence')
    print(f"⚙️  Configuration applied for {expected_persons} persons")

    # Step 1: Extract person crops
    print("\n📹 Step 1: Video Processing")
    crops, detection_stats = extract_person_crops(
        video_path, confidence=0.9, frame_skip=frame_skip, max_frames=max_frames
    )

    if not crops:
        print("❌ No person crops extracted. Exiting.")
        return

    print(f"   - Total crops extracted: {len(crops)}")

    # Step 2: Feature extraction
    print("\n🔍 Step 2: Feature Extraction")
    extractor = OSNetReID()
    features, metadata = extract_features(crops, extractor)

    print(f"   - Features extracted: {features.shape}")

    # Step 3: Pose2ID clustering
    print("\n🎯 Step 3: Pose2ID Clustering")
    labels, features_nfc, metrics = pose2id_clustering(
        features, k1=k1, k2=k2, cluster_method=cluster_method, n_clusters=expected_persons
    )

    # Step 4: Compute ID² metric
    print("\n📊 Step 4: ID² Metric Computation")
    id2_scores = ID2(features_nfc, np.array(labels))
    global_density = np.mean(id2_scores)

    print(f"   - Global Identity Density (ID²): {global_density:.3f}")
    print(f"   - Per-cluster ID² scores: {dict(zip(set(labels), id2_scores))}")

    # Step 5: Visualization
    print("\n🎨 Step 5: Visualization")
    viz_path = create_pose2id_visualization(
        features, features_nfc, labels, metadata, output_dir
    )
    print(f"   - Visualization saved: {viz_path}")

    # Step 6: Save results
    print("\n💾 Step 6: Save Results")

    # Organize results by cluster
    clusters = defaultdict(list)
    for label, meta in zip(labels, metadata):
        clusters[int(label)].append(meta['frame'])

    # Comprehensive results
    results = {
        'video': str(video_path),
        'method': 'Pose2ID_NFC',
        'parameters': {
            'k1': k1,
            'k2': k2,
            'cluster_method': cluster_method,
            'expected_persons': expected_persons
        },
        'results': {
            'clusters_found': metrics['n_clusters'],
            'cluster_sizes': metrics['cluster_sizes'],
            'silhouette_score': metrics['silhouette_score'],
            'nfc_improvement': metrics['nfc_improvement'],
            'global_id2_density': float(global_density),
            'per_cluster_id2': {str(k): float(v) for k, v in zip(set(labels), id2_scores)}
        },
        'clusters': {
            str(k): {
                'size': len(v),
                'frame_range': [int(min(v)), int(max(v))] if v else [0, 0]
            } for k, v in clusters.items()
        },
        'detection_stats': dict(detection_stats)
    }

    results_path = output_dir / 'pose2id_paper_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"   - Results saved: {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("📋 POSE2ID CLUSTERING SUMMARY")
    print("=" * 70)
    print(
        f"🎯 Clusters: {metrics['n_clusters']} (expected: {expected_persons})")
    print(f"📊 Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"🔄 NFC Improvement: {metrics['nfc_improvement']:.3f}")
    print(f"🆔 Global ID² Density: {global_density:.3f}")
    print(f"📈 Cluster Distribution: {metrics['cluster_sizes']}")

    if metrics['n_clusters'] == expected_persons:
        print("✅ SUCCESS: Achieved expected number of clusters!")
    else:
        print(
            f"⚠️  Note: Found {metrics['n_clusters']} clusters, expected {expected_persons}")

    print("=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Pose2ID Paper Implementation - Feature Centralization Clustering')
    parser.add_argument(
        '--video', default='data/raw/video3.mp4', help='Input video path')
    parser.add_argument(
        '--output-dir', default='output/pose2id_paper', help='Output directory')
    parser.add_argument('--expected-persons', type=int,
                        default=5, help='Expected number of persons')
    parser.add_argument('--k1', type=int, default=4,
                        help='NFC parameter k1 (nearest neighbors)')
    parser.add_argument('--k2', type=int, default=4,
                        help='NFC parameter k2 (mutual neighbors)')
    parser.add_argument(
        '--cluster-method', choices=['kmeans', 'dbscan'], default='kmeans', help='Clustering method')
    parser.add_argument('--frame-skip', type=int, default=2,
                        help='Process every N frames')
    parser.add_argument('--max-frames', type=int,
                        default=None, help='Maximum frames to process')

    args = parser.parse_args()

    main(
        video_path=args.video,
        output_dir=args.output_dir,
        expected_persons=args.expected_persons,
        k1=args.k1,
        k2=args.k2,
        cluster_method=args.cluster_method,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames
    )
