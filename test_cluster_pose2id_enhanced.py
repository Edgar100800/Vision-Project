#!/usr/bin/env python3
"""
Enhanced Pose2ID Implementation with Visual Centroids and Extended Features
Based on: "From Poses to Identity: Training-Free Person Re-Identification via Feature Centralization"
Enhanced with visual centroid generation and 15-feature expansion
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
from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches

# Add project paths
sys.path.append('.')
sys.path.append('src')


# --------------------------------------------------------------------------------------
# Enhanced NFC with 15-feature expansion
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


def enhanced_feature_extraction(features, metadata):
    """
    Expand features to 15 dimensions for better clustering

    Args:
        features: Original features [N, 512]
        metadata: Metadata with frame info, confidence, etc.

    Returns:
        Enhanced features [N, 15]
    """
    print("🔧 Expanding features to 15 dimensions...")

    # 1. PCA to reduce to 10 main components
    pca = PCA(n_components=10, random_state=42)
    pca_features = pca.fit_transform(features)

    # 2. Add 5 additional engineered features
    additional_features = []

    for i, meta in enumerate(metadata):
        # Feature 11: Temporal position (normalized frame number)
        max_frame = max(m['frame'] for m in metadata)
        temporal_pos = meta['frame'] / max_frame

        # Feature 12: Confidence score
        confidence = meta['conf']

        # Feature 13: Temporal density (frames per second around this detection)
        frame_window = 60  # 2 seconds at 30fps
        nearby_frames = sum(1 for m in metadata
                            if abs(m['frame'] - meta['frame']) <= frame_window)
        temporal_density = nearby_frames / len(metadata)

        # Feature 14: Feature magnitude (L2 norm of original features)
        feature_magnitude = np.linalg.norm(features[i])

        # Feature 15: Relative position in video (early/middle/late)
        relative_position = 0.0
        if temporal_pos < 0.33:
            relative_position = 0.0  # Early
        elif temporal_pos < 0.66:
            relative_position = 0.5  # Middle
        else:
            relative_position = 1.0  # Late

        additional_features.append([
            temporal_pos, confidence, temporal_density,
            feature_magnitude, relative_position
        ])

    additional_features = np.array(additional_features)

    # Normalize additional features
    additional_features = (additional_features - additional_features.mean(axis=0)
                           ) / (additional_features.std(axis=0) + 1e-8)

    # Combine PCA features with additional features
    enhanced_features = np.hstack([pca_features, additional_features])

    print(f"   - Original features: {features.shape}")
    print(f"   - Enhanced features: {enhanced_features.shape}")
    print(
        f"   - PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    return enhanced_features, pca


def NFC_enhanced(feat: torch.tensor, k1=6, k2=6, alpha=0.5):
    """
    Enhanced Neighbor Feature Centralization with adaptive weighting

    Args:
        feat: Feature tensor [N, 15]
        k1: Number of nearest neighbors to consider
        k2: Number of mutual neighbors required
        alpha: Weighting factor for neighbor contribution

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

    # Find mutual top-k neighbors with distance weighting
    mutual_topk_list = []
    mutual_weights_list = []

    for i in range(rank.size(0)):
        mutual_list = []
        weight_list = []
        for j_idx, j in enumerate(rank[i]):
            if i in rank[j][:k2]:
                mutual_list.append(j.item())
                # Weight by inverse distance (closer neighbors have more influence)
                weight = 1.0 / (val[i][j_idx] + 1e-8)
                weight_list.append(weight)

        # Normalize weights
        if weight_list:
            weight_list = np.array(weight_list)
            weight_list = weight_list / weight_list.sum()

        mutual_topk_list.append(mutual_list)
        mutual_weights_list.append(weight_list)

    # Enhanced feature centralization with adaptive weighting
    feat_copy = feat.clone()
    for i in range(rank.size(0)):
        if mutual_topk_list[i]:
            # Weighted sum of mutual neighbors
            neighbor_feats = feat_copy[mutual_topk_list[i]]
            weights = torch.tensor(
                mutual_weights_list[i], device=device).unsqueeze(1)
            weighted_neighbors = (neighbor_feats * weights).sum(dim=0)

            # Adaptive combination with original feature
            feat[i] = (1 - alpha) * feat[i] + alpha * weighted_neighbors

    return feat

# --------------------------------------------------------------------------------------
# Visual Centroid Generation
# --------------------------------------------------------------------------------------


def create_visual_centroids(clusters, crops_data, metadata, output_dir):
    """
    Generate visual centroids for each cluster

    Args:
        clusters: Dictionary of cluster assignments
        crops_data: Original crop images
        metadata: Metadata with indices
        output_dir: Output directory for centroids

    Returns:
        Dictionary of centroid image paths
    """
    print("🎨 Generating visual centroids...")
    output_dir = Path(output_dir)
    centroids_dir = output_dir / 'centroids'
    centroids_dir.mkdir(parents=True, exist_ok=True)

    centroid_paths = {}

    for cluster_id, frame_list in clusters.items():
        if not frame_list:
            continue

        print(
            f"   - Processing cluster {cluster_id} ({len(frame_list)} samples)")

        # Get crop indices for this cluster
        cluster_indices = []
        for meta_idx, meta in enumerate(metadata):
            if meta['frame'] in frame_list:
                cluster_indices.append(meta['idx'])

        if not cluster_indices:
            continue

        # Select representative images (up to 9 for 3x3 grid)
        n_representatives = min(9, len(cluster_indices))

        # Smart selection: spread across time + highest confidence
        cluster_meta = [(idx, metadata[i])
                        for i, idx in enumerate(cluster_indices)]
        # Sort by confidence
        cluster_meta.sort(key=lambda x: x[1]['conf'], reverse=True)

        # Take top confidence images, but ensure temporal spread
        selected_indices = []
        selected_frames = []

        for idx, meta in cluster_meta:
            if len(selected_indices) >= n_representatives:
                break

            # Check temporal spread (don't select too many from same time period)
            frame = meta['frame']
            # 1 second apart
            too_close = any(abs(frame - sf) < 30 for sf in selected_frames)

            if not too_close or len(selected_indices) < 3:  # Always take at least 3
                selected_indices.append(idx)
                selected_frames.append(frame)

        # Create centroid image
        centroid_path = create_cluster_centroid_image(
            selected_indices, crops_data, cluster_id, centroids_dir
        )

        centroid_paths[cluster_id] = str(centroid_path)

    return centroid_paths


def create_cluster_centroid_image(indices, crops_data, cluster_id, output_dir):
    """Create a single centroid image for a cluster"""

    # Load and resize crops
    crop_size = 128
    crops = []

    for idx in indices:
        if idx < len(crops_data):
            crop = crops_data[idx]['crop']
            crop_resized = cv2.resize(crop, (crop_size, crop_size))
            crops.append(crop_resized)

    if not crops:
        return None

    # Arrange in grid
    n_crops = len(crops)
    grid_size = int(np.ceil(np.sqrt(n_crops)))

    # Create grid image
    grid_img = np.zeros(
        (grid_size * crop_size, grid_size * crop_size, 3), dtype=np.uint8)

    for i, crop in enumerate(crops):
        row = i // grid_size
        col = i % grid_size
        y_start = row * crop_size
        x_start = col * crop_size
        grid_img[y_start:y_start + crop_size,
                 x_start:x_start + crop_size] = crop

    # Add cluster information
    grid_img = add_cluster_info(grid_img, cluster_id, len(indices))

    # Save centroid image
    centroid_path = output_dir / f'cluster_{cluster_id}_centroid.jpg'
    cv2.imwrite(str(centroid_path), grid_img)

    return centroid_path


def add_cluster_info(image, cluster_id, n_samples):
    """Add cluster information to centroid image"""

    # Convert to PIL for text rendering
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Add text
    text = f"Cluster {cluster_id}\n{n_samples} samples"
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --------------------------------------------------------------------------------------
# Enhanced Clustering with 15 features
# --------------------------------------------------------------------------------------


def enhanced_pose2id_clustering(features, metadata, k1=6, k2=6, cluster_method='kmeans', n_clusters=5):
    """
    Enhanced Pose2ID clustering with 15-feature expansion

    Args:
        features: Input features [N, 512]
        metadata: Metadata with frame info
        k1, k2: NFC parameters
        cluster_method: 'kmeans' or 'dbscan'
        n_clusters: Number of expected clusters

    Returns:
        labels, centralized_features, metrics, enhanced_features
    """
    print(f"🔬 Enhanced Pose2ID Clustering Pipeline")
    print(f"   - Input features: {features.shape}")
    print(f"   - NFC params: k1={k1}, k2={k2}")
    print(f"   - Clustering method: {cluster_method}")

    # Step 1: Expand features to 15 dimensions
    enhanced_features, pca = enhanced_feature_extraction(features, metadata)

    # Step 2: Apply Enhanced NFC
    print("📐 Applying Enhanced NFC...")
    features_tensor = torch.tensor(enhanced_features).float()
    features_nfc = NFC_enhanced(features_tensor, k1=k1, k2=k2, alpha=0.6)
    features_nfc = features_nfc.numpy()

    # Step 3: L2 Normalization
    print("🔄 L2 Normalization...")
    features_nfc = features_nfc / \
        (np.linalg.norm(features_nfc, axis=1, keepdims=True) + 1e-8)

    # Step 4: Enhanced Clustering
    print(f"🎯 Enhanced Clustering with {cluster_method}...")
    if cluster_method == 'kmeans':
        # Use K-means++ initialization for better centroids
        clusterer = KMeans(n_clusters=n_clusters,
                           random_state=42, n_init=20, init='k-means++')
        labels = clusterer.fit_predict(features_nfc)
    elif cluster_method == 'dbscan':
        # Adaptive parameters for 15D space
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=min(6, len(features_nfc)//10))
        neighbors_fit = neighbors.fit(features_nfc)
        distances, indices = neighbors_fit.kneighbors(features_nfc)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        eps = np.percentile(distances, 75)  # More conservative eps for 15D

        clusterer = DBSCAN(eps=eps, min_samples=max(4, len(features_nfc)//40))
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

    # Step 5: Compute enhanced metrics
    print("📊 Computing enhanced metrics...")
    metrics = {}

    # Silhouette Score
    if len(set(labels)) > 1:
        metrics['silhouette_score'] = silhouette_score(features_nfc, labels)
        original_silhouette = silhouette_score(enhanced_features, labels)
        metrics['nfc_improvement'] = metrics['silhouette_score'] - \
            original_silhouette
    else:
        metrics['silhouette_score'] = 0.0
        metrics['nfc_improvement'] = 0.0

    # Cluster statistics
    unique_labels = set(labels)
    metrics['n_clusters'] = len(unique_labels)
    metrics['cluster_sizes'] = {int(label): int(
        np.sum(labels == label)) for label in unique_labels}

    # Feature quality metrics
    metrics['feature_expansion_ratio'] = enhanced_features.shape[1] / \
        features.shape[1]
    metrics['pca_explained_variance'] = float(
        pca.explained_variance_ratio_.sum())

    print(f"✅ Enhanced clustering completed:")
    print(f"   - Clusters found: {metrics['n_clusters']}")
    print(f"   - Silhouette score: {metrics['silhouette_score']:.3f}")
    print(f"   - NFC improvement: {metrics['nfc_improvement']:.3f}")
    print(
        f"   - Feature expansion: {features.shape[1]} → {enhanced_features.shape[1]}")

    return labels, features_nfc, metrics, enhanced_features

# --------------------------------------------------------------------------------------
# Enhanced ID² Metric
# --------------------------------------------------------------------------------------


def enhanced_ID2(features, pids, feature_weights=None):
    """
    Enhanced ID² Metric with feature weighting

    Args:
        features: Feature matrix [N, 15]
        pids: Person IDs [N]
        feature_weights: Optional weights for different features

    Returns:
        Enhanced identity density scores
    """
    if feature_weights is None:
        feature_weights = np.ones(features.shape[1])

    unique_pids = np.unique(pids)
    densities = []

    for pid in unique_pids:
        mask = pids == pid
        if np.sum(mask) < 2:
            densities.append(0.0)
            continue

        pid_features = features[mask]

        # Weighted intra-class distances
        intra_distances = []
        for i in range(len(pid_features)):
            for j in range(i+1, len(pid_features)):
                diff = pid_features[i] - pid_features[j]
                weighted_dist = np.sqrt(np.sum((diff ** 2) * feature_weights))
                intra_distances.append(weighted_dist)

        # Weighted inter-class distances
        inter_distances = []
        other_features = features[~mask]
        for feat in pid_features:
            for other_feat in other_features:
                diff = feat - other_feat
                weighted_dist = np.sqrt(np.sum((diff ** 2) * feature_weights))
                inter_distances.append(weighted_dist)

        # Enhanced identity density
        if intra_distances:
            avg_intra = np.mean(intra_distances)
            avg_inter = np.mean(inter_distances) if inter_distances else 1.0
            density = avg_inter / (avg_intra + 1e-8)
        else:
            density = 0.0

        densities.append(density)

    return np.array(densities)

# --------------------------------------------------------------------------------------
# Video Processing (same as before)
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
# Enhanced Visualization
# --------------------------------------------------------------------------------------


def create_enhanced_visualization(features_original, features_enhanced, features_nfc, labels, metadata, output_dir):
    """Create comprehensive visualization with all feature stages"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate PCA for each feature type (different dimensions)
    pca_original = PCA(n_components=2, random_state=42)
    pca_enhanced = PCA(n_components=2, random_state=42)
    pca_nfc = PCA(n_components=2, random_state=42)
    
    # Transform each feature type with its own PCA
    original_2d = pca_original.fit_transform(features_original)
    enhanced_2d = pca_enhanced.fit_transform(features_enhanced)
    nfc_2d = pca_nfc.fit_transform(features_nfc)

    # Create 3-panel comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    unique_labels = sorted(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    feature_stages = [
        (original_2d, 'Original Features (512D)', axes[0]),
        (enhanced_2d, 'Enhanced Features (15D)', axes[1]),
        (nfc_2d, 'NFC Features (15D)', axes[2])
    ]

    for features_2d, title, ax in feature_stages:
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors[i]], label=f'Person {label}', alpha=0.7, s=30)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = output_dir / 'enhanced_pose2id_visualization.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    return viz_path

# --------------------------------------------------------------------------------------
# Main Enhanced Pipeline
# --------------------------------------------------------------------------------------


def main(video_path: str, output_dir: str, expected_persons: int = 5,
         k1: int = 6, k2: int = 6, cluster_method: str = 'kmeans',
         frame_skip: int = 2, max_frames: int = None):
    """
    Enhanced Pose2ID clustering pipeline with visual centroids and 15 features
    """

    print("🚀 Enhanced Pose2ID Implementation - Visual Centroids + 15 Features")
    print("=" * 80)

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

    # Step 3: Enhanced Pose2ID clustering
    print("\n🎯 Step 3: Enhanced Pose2ID Clustering")
    labels, features_nfc, metrics, enhanced_features = enhanced_pose2id_clustering(
        features, metadata, k1=k1, k2=k2, cluster_method=cluster_method, n_clusters=expected_persons
    )

    # Step 4: Enhanced ID² metric
    print("\n📊 Step 4: Enhanced ID² Metric")
    id2_scores = enhanced_ID2(features_nfc, np.array(labels))
    global_density = np.mean(id2_scores)

    print(f"   - Global Enhanced ID² Density: {global_density:.3f}")

    # Step 5: Generate visual centroids
    print("\n🎨 Step 5: Visual Centroid Generation")

    # Organize results by cluster
    clusters = defaultdict(list)
    for label, meta in zip(labels, metadata):
        clusters[int(label)].append(meta['frame'])

    centroid_paths = create_visual_centroids(
        clusters, crops, metadata, output_dir)
    print(f"   - Generated {len(centroid_paths)} visual centroids")

    # Step 6: Enhanced visualization
    print("\n🎨 Step 6: Enhanced Visualization")
    viz_path = create_enhanced_visualization(
        features, enhanced_features, features_nfc, labels, metadata, output_dir
    )
    print(f"   - Enhanced visualization saved: {viz_path}")

    # Step 7: Save comprehensive results
    print("\n💾 Step 7: Save Enhanced Results")

    # Comprehensive results
    results = {
        'video': str(video_path),
        'method': 'Enhanced_Pose2ID_NFC',
        'parameters': {
            'k1': k1,
            'k2': k2,
            'cluster_method': cluster_method,
            'expected_persons': expected_persons,
            'features_expanded': '15D'
        },
        'results': {
            'clusters_found': metrics['n_clusters'],
            'cluster_sizes': metrics['cluster_sizes'],
            'silhouette_score': metrics['silhouette_score'],
            'nfc_improvement': metrics['nfc_improvement'],
            'global_enhanced_id2_density': float(global_density),
            'per_cluster_id2': {str(k): float(v) for k, v in zip(set(labels), id2_scores)},
            'feature_expansion_ratio': metrics['feature_expansion_ratio'],
            'pca_explained_variance': metrics['pca_explained_variance']
        },
        'clusters': {
            str(k): {
                'size': len(v),
                'frame_range': [int(min(v)), int(max(v))] if v else [0, 0],
                'centroid_image': centroid_paths.get(k, None)
            } for k, v in clusters.items()
        },
        'detection_stats': dict(detection_stats),
        'centroid_paths': centroid_paths
    }

    results_path = output_dir / 'enhanced_pose2id_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"   - Enhanced results saved: {results_path}")

    # Enhanced Summary
    print("\n" + "=" * 80)
    print("📋 ENHANCED POSE2ID CLUSTERING SUMMARY")
    print("=" * 80)
    print(
        f"🎯 Clusters: {metrics['n_clusters']} (expected: {expected_persons})")
    print(f"📊 Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"🔄 NFC Improvement: {metrics['nfc_improvement']:.3f}")
    print(f"🆔 Enhanced ID² Density: {global_density:.3f}")
    print(
        f"📈 Feature Expansion: 512D → 15D ({metrics['feature_expansion_ratio']:.1f}x)")
    print(f"🎨 Visual Centroids: {len(centroid_paths)} generated")
    print(f"📊 Cluster Distribution: {metrics['cluster_sizes']}")
    print(f"🔍 PCA Explained Variance: {metrics['pca_explained_variance']:.3f}")

    if metrics['n_clusters'] == expected_persons:
        print("✅ SUCCESS: Achieved expected number of clusters with enhanced features!")
    else:
        print(
            f"⚠️  Note: Found {metrics['n_clusters']} clusters, expected {expected_persons}")

    print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Enhanced Pose2ID - Visual Centroids + 15 Features')
    parser.add_argument(
        '--video', default='data/raw/video3.mp4', help='Input video path')
    parser.add_argument(
        '--output-dir', default='output/enhanced_pose2id', help='Output directory')
    parser.add_argument('--expected-persons', type=int,
                        default=5, help='Expected number of persons')
    parser.add_argument('--k1', type=int, default=6,
                        help='NFC parameter k1 (nearest neighbors)')
    parser.add_argument('--k2', type=int, default=6,
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
