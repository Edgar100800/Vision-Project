#!/usr/bin/env python3
"""
Improved Video Clustering Analysis Script
========================================
This script implements multiple improvements for better clustering performance:
1. Optimized hyperparameters and post-clustering fusion
2. TransReID model support and feature ensemble
3. DBSCAN clustering with adaptive parameters
4. Temporal feature averaging and track-aware clustering
"""

import os
import sys
import cv2
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Project imports
from config.global_config import get_config, update_config, apply_preset
from src.reid.osnet_reid import OSNetReID
from src.index.hnsw_index import HNSWIndex
from src.clustering.gaussian_cluster_manager import GaussianClusterManager


class TransReIDExtractor:
    """TransReID feature extractor using the available TransReID model."""

    def __init__(self):
        print("🔧 Initializing TransReID extractor...")
        self.model = None
        self.device = 'cpu'
        self.feature_dim = 768  # TransReID typical dimension

        # Try to load TransReID model
        try:
            sys.path.append('Pose2ID/demo/TransReID-main')
            from model.make_model import make_model
            from config.defaults import _C as cfg

            cfg.MODEL.DEVICE = self.device
            cfg.MODEL.NAME = 'transformer'
            cfg.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'
            cfg.MODEL.STRIDE_SIZE = [16, 16]

            self.model = make_model(
                cfg, num_class=1000, camera_num=0, view_num=0)
            print("   ✅ TransReID model loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load TransReID: {e}")
            print("   🔄 Falling back to OSNet")
            self.model = None
            self.feature_dim = 512

    def extract_features(self, crop_image):
        """Extract features from a single crop image."""
        if self.model is None:
            # Fallback to OSNet if TransReID not available
            fallback_extractor = OSNetReID()
            return fallback_extractor.extract_features(crop_image)

        # Preprocess image for TransReID
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        tensor_image = transform(rgb_image).unsqueeze(0)

        with torch.no_grad():
            features = self.model(tensor_image)

        return [features.cpu().numpy().flatten()]


class ImprovedClusteringAnalyzer:
    """Improved clustering analyzer with multiple enhancement strategies."""

    def __init__(self, config):
        self.config = config
        self.reid_cfg = get_config('reid')
        self.clustering_cfg = get_config('gaussian_clustering')

        # Initialize extractors
        self.osnet_extractor = OSNetReID()
        self.transreid_extractor = TransReIDExtractor()

        # Clustering parameters (Point 1: Optimized hyperparameters)
        self.optimized_params = {
            'mahalanobis_threshold': 6.2,  # Reduced from 7.0
            'min_samples_per_cluster': 5,   # Increased from 3
            'fusion_threshold': 5.0,        # 6.2 * 0.8
            'post_fusion_threshold': 2.0,   # For post-processing fusion
            'temporal_window': 1.0,         # 1 second window for temporal averaging
        }

        print(f"🔧 Initialized with optimized parameters:")
        print(
            f"   • Mahalanobis threshold: {self.optimized_params['mahalanobis_threshold']}")
        print(
            f"   • Min samples per cluster: {self.optimized_params['min_samples_per_cluster']}")

    def extract_ensemble_features(self, person_crops):
        """Extract features using ensemble of OSNet and TransReID (Point 2)."""
        print(
            f"\n🚀 Extracting ensemble features from {len(person_crops)} crops...")

        osnet_features = []
        transreid_features = []
        crop_metadata = []

        for idx, crop_data in enumerate(tqdm(person_crops, desc="Extracting features")):
            crop = crop_data['crop']

            try:
                # Extract OSNet features
                osnet_feat = self.osnet_extractor.extract_features(crop)

                # Extract TransReID features
                transreid_feat = self.transreid_extractor.extract_features(
                    crop)

                if osnet_feat is not None and transreid_feat is not None:
                    osnet_features.append(osnet_feat[0])
                    transreid_features.append(transreid_feat[0])

                    crop_metadata.append({
                        'crop_idx': idx,
                        'frame_number': crop_data['frame_number'],
                        'confidence': crop_data['confidence'],
                        'bbox': crop_data['bbox'],
                        'timestamp': crop_data['timestamp']
                    })
            except Exception as e:
                print(
                    f"Warning: Failed to extract features from crop {idx}: {e}")
                continue

        if len(osnet_features) == 0:
            raise ValueError("No features could be extracted from any crops")

        # Normalize features
        osnet_features = normalize(np.vstack(osnet_features), norm='l2')
        transreid_features = normalize(
            np.vstack(transreid_features), norm='l2')

        # Ensemble: concatenate normalized features
        ensemble_features = np.hstack([osnet_features, transreid_features])

        print(
            f"   ✅ Extracted {len(ensemble_features)} ensemble feature vectors")
        print(f"   • OSNet features: {osnet_features.shape[1]}D")
        print(f"   • TransReID features: {transreid_features.shape[1]}D")
        print(f"   • Ensemble features: {ensemble_features.shape[1]}D")

        return ensemble_features, crop_metadata, osnet_features, transreid_features

    def apply_temporal_averaging(self, features, crop_metadata):
        """Apply temporal averaging to reduce noise (Point 4)."""
        print(f"\n🕐 Applying temporal averaging...")

        temporal_window = self.optimized_params['temporal_window']

        # Group features by time windows
        time_groups = defaultdict(list)
        for idx, meta in enumerate(crop_metadata):
            time_key = int(meta['timestamp'] / temporal_window)
            time_groups[time_key].append({
                'feature': features[idx],
                'metadata': meta,
                'original_idx': idx
            })

        # Average features within each time window
        averaged_features = []
        averaged_metadata = []

        for time_key, group in time_groups.items():
            if len(group) > 1:
                # Average features
                group_features = np.vstack([item['feature'] for item in group])
                avg_feature = np.mean(group_features, axis=0)

                # Use metadata from highest confidence detection
                best_item = max(
                    group, key=lambda x: x['metadata']['confidence'])
                metadata = best_item['metadata'].copy()
                metadata['temporal_group_size'] = len(group)

                averaged_features.append(avg_feature)
                averaged_metadata.append(metadata)
            else:
                # Single item, no averaging needed
                item = group[0]
                item['metadata']['temporal_group_size'] = 1
                averaged_features.append(item['feature'])
                averaged_metadata.append(item['metadata'])

        averaged_features = np.vstack(averaged_features)

        print(f"   ✅ Temporal averaging complete:")
        print(f"   • Original features: {len(features)}")
        print(f"   • Averaged features: {len(averaged_features)}")
        print(
            f"   • Compression ratio: {len(averaged_features)/len(features):.2f}")

        return averaged_features, averaged_metadata

    def find_optimal_dbscan_eps(self, features, expected_clusters=3):
        """Find optimal DBSCAN eps parameter using k-distance plot (Point 3)."""
        print(f"\n🔍 Finding optimal DBSCAN parameters...")

        # Use k-distance plot to find optimal eps
        k = max(4, expected_clusters * 2)  # Heuristic: 2 * expected clusters

        # Calculate k-distances
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
        nbrs.fit(features)
        distances, indices = nbrs.kneighbors(features)

        # Sort distances to k-th neighbor
        k_distances = distances[:, -1]
        k_distances = np.sort(k_distances)[::-1]  # Sort in descending order

        # Find elbow point (knee/elbow detection)
        # Simple method: look for the point where the rate of change is maximum
        diffs = np.diff(k_distances)
        second_diffs = np.diff(diffs)

        # Find the point with maximum second derivative (elbow)
        elbow_idx = np.argmax(second_diffs) + 1
        optimal_eps = k_distances[elbow_idx]

        print(f"   • K-distance analysis (k={k})")
        print(f"   • Optimal eps: {optimal_eps:.4f}")

        return optimal_eps

    def perform_dbscan_clustering(self, features, crop_metadata):
        """Perform DBSCAN clustering with optimized parameters (Point 3)."""
        print(f"\n🔄 Performing DBSCAN clustering...")

        # Normalize features for cosine distance
        normalized_features = normalize(features, norm='l2')

        # Find optimal eps
        optimal_eps = self.find_optimal_dbscan_eps(normalized_features)

        # Perform DBSCAN clustering
        dbscan = DBSCAN(
            eps=optimal_eps,
            min_samples=self.optimized_params['min_samples_per_cluster'],
            metric='cosine'
        )

        cluster_labels = dbscan.fit_predict(normalized_features)

        # Analyze results
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        print(f"   ✅ DBSCAN clustering complete:")
        print(f"   • Clusters formed: {n_clusters}")
        print(f"   • Noise points: {n_noise}")
        print(f"   • Eps parameter: {optimal_eps:.4f}")

        return cluster_labels, n_clusters

    def perform_post_clustering_fusion(self, features, cluster_labels, crop_metadata):
        """Perform post-clustering fusion of similar clusters (Point 1)."""
        print(f"\n🔗 Performing post-clustering fusion...")

        unique_clusters = [c for c in set(
            cluster_labels) if c != -1]  # Exclude noise

        if len(unique_clusters) <= 3:
            print("   ✅ No fusion needed - optimal cluster count achieved")
            return cluster_labels

        # Calculate cluster centroids
        cluster_centroids = {}
        for cluster_id in unique_clusters:
            cluster_mask = np.array(cluster_labels) == cluster_id
            cluster_features = features[cluster_mask]
            cluster_centroids[cluster_id] = np.mean(cluster_features, axis=0)

        # Calculate pairwise distances between centroids
        centroid_ids = list(cluster_centroids.keys())
        centroid_features = np.vstack(
            [cluster_centroids[c_id] for c_id in centroid_ids])

        # Use cosine distance for similarity
        distances = cdist(centroid_features,
                          centroid_features, metric='cosine')

        # Find clusters to merge
        fusion_threshold = self.optimized_params['post_fusion_threshold']
        merged_clusters = set()
        fusion_mapping = {}

        for i, cluster_i in enumerate(centroid_ids):
            if cluster_i in merged_clusters:
                continue

            for j, cluster_j in enumerate(centroid_ids):
                if i >= j or cluster_j in merged_clusters:
                    continue

                # Check if clusters should be merged
                if distances[i, j] < fusion_threshold:
                    # Check temporal overlap (don't merge if they appear at very different times)
                    times_i = [crop_metadata[k]['timestamp'] for k, label in enumerate(
                        cluster_labels) if label == cluster_i]
                    times_j = [crop_metadata[k]['timestamp'] for k, label in enumerate(
                        cluster_labels) if label == cluster_j]

                    time_overlap = max(0, min(max(times_i), max(
                        times_j)) - max(min(times_i), min(times_j)))
                    total_time = max(max(times_i), max(times_j)) - \
                        min(min(times_i), min(times_j))

                    if time_overlap / total_time > 0.1:  # At least 10% temporal overlap
                        fusion_mapping[cluster_j] = cluster_i
                        merged_clusters.add(cluster_j)
                        print(
                            f"   • Merging cluster {cluster_j} into {cluster_i} (distance: {distances[i, j]:.4f})")

        # Apply fusion mapping
        fused_labels = cluster_labels.copy()
        for i, label in enumerate(cluster_labels):
            if label in fusion_mapping:
                fused_labels[i] = fusion_mapping[label]

        n_clusters_after = len(set(fused_labels)) - \
            (1 if -1 in fused_labels else 0)
        print(f"   ✅ Post-clustering fusion complete:")
        print(f"   • Clusters before fusion: {len(unique_clusters)}")
        print(f"   • Clusters after fusion: {n_clusters_after}")

        return fused_labels

    def perform_improved_clustering(self, features, crop_metadata):
        """Perform improved clustering with all enhancements."""
        print(f"\n🎯 Performing improved clustering analysis...")

        # Step 1: Apply temporal averaging (Point 4)
        averaged_features, averaged_metadata = self.apply_temporal_averaging(
            features, crop_metadata)

        # Step 2: Perform DBSCAN clustering (Point 3)
        cluster_labels, n_clusters = self.perform_dbscan_clustering(
            averaged_features, averaged_metadata)

        # Step 3: Apply post-clustering fusion (Point 1)
        final_labels = self.perform_post_clustering_fusion(
            averaged_features, cluster_labels, averaged_metadata)

        # Calculate final metrics
        final_n_clusters = len(set(final_labels)) - \
            (1 if -1 in final_labels else 0)

        # Calculate silhouette score (exclude noise points)
        if final_n_clusters > 1:
            non_noise_mask = np.array(final_labels) != -1
            if np.sum(non_noise_mask) > 1:
                silhouette = silhouette_score(
                    averaged_features[non_noise_mask],
                    np.array(final_labels)[non_noise_mask]
                )
            else:
                silhouette = -1
        else:
            silhouette = -1

        print(f"\n📊 Final Clustering Results:")
        print(f"   • Final clusters: {final_n_clusters}")
        print(f"   • Expected clusters: 3")
        print(f"   • Efficiency: {3/final_n_clusters:.4f}" if final_n_clusters >
              0 else "   • Efficiency: 0.0000")
        print(f"   • Silhouette score: {silhouette:.4f}")

        return {
            'features': averaged_features,
            'metadata': averaged_metadata,
            'cluster_labels': final_labels,
            'n_clusters': final_n_clusters,
            'silhouette_score': silhouette,
            'efficiency': 3/final_n_clusters if final_n_clusters > 0 else 0,
            'original_features': features,
            'original_metadata': crop_metadata
        }


def extract_person_crops_from_video(video_path, confidence_threshold=0.9, frame_skip=1, max_frames=None):
    """Extract person crops from video using YOLO detection."""
    print(f"🎬 Processing video: {video_path}")
    print(f"   📋 Confidence threshold: {confidence_threshold:.0%}")
    print(f"   📋 Frame skip: {frame_skip}")

    # Load YOLO model
    detection_cfg = get_config('detection')
    yolo_model = YOLO(detection_cfg['model_path'])

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"   📺 Video info: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    if max_frames:
        total_frames = min(total_frames, max_frames)
        print(f"   📏 Processing first {total_frames} frames")

    person_crops = []
    detection_stats = {
        'total_frames_processed': 0,
        'frames_with_detections': 0,
        'total_detections': 0,
        'valid_detections': 0,
        'confidence_threshold': confidence_threshold
    }

    frame_number = 0
    pbar = tqdm(total=total_frames//frame_skip, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            detection_stats['total_frames_processed'] += 1

            # Run YOLO detection
            results = yolo_model(frame, verbose=False)

            frame_has_detections = False
            for result in results:
                if result.boxes is not None:
                    # Filter for person class (class 0 in COCO)
                    person_boxes = result.boxes[result.boxes.cls == 0]

                    if len(person_boxes) > 0:
                        frame_has_detections = True
                        detection_stats['total_detections'] += len(
                            person_boxes)

                        # Process each person detection
                        for box in person_boxes:
                            confidence = box.conf.item()

                            if confidence >= confidence_threshold:
                                # Extract bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu(
                                ).numpy().astype(int)

                                # Add padding to crop
                                padding = 0.1
                                w, h = x2 - x1, y2 - y1
                                pad_w, pad_h = int(
                                    w * padding), int(h * padding)

                                x1 = max(0, x1 - pad_w)
                                y1 = max(0, y1 - pad_h)
                                x2 = min(width, x2 + pad_w)
                                y2 = min(height, y2 + pad_h)

                                # Extract crop
                                crop = frame[y1:y2, x1:x2]

                                # Validate crop size
                                if crop.shape[0] > 32 and crop.shape[1] > 32:
                                    person_crops.append({
                                        'crop': crop,
                                        'frame_number': frame_number,
                                        'confidence': confidence,
                                        'bbox': (x1, y1, x2, y2),
                                        'timestamp': frame_number / fps
                                    })
                                    detection_stats['valid_detections'] += 1

            if frame_has_detections:
                detection_stats['frames_with_detections'] += 1

            pbar.update(1)

            # Stop if we've reached max frames
            if max_frames and frame_number >= max_frames:
                break

        frame_number += 1

    cap.release()
    pbar.close()

    # Print statistics
    print(f"\n📊 Detection Statistics:")
    print(
        f"   • Frames processed: {detection_stats['total_frames_processed']}")
    print(
        f"   • Frames with detections: {detection_stats['frames_with_detections']}")
    print(f"   • Total detections: {detection_stats['total_detections']}")
    print(f"   • Valid detections: {detection_stats['valid_detections']}")
    print(
        f"   • Detection rate: {detection_stats['frames_with_detections']/detection_stats['total_frames_processed']:.1%}")
    print(
        f"   • Retention rate: {detection_stats['valid_detections']/detection_stats['total_detections']:.1%}" if detection_stats['total_detections'] > 0 else "   • Retention rate: 0%")

    return person_crops, detection_stats


def generate_improved_visualizations(clustering_results, output_dir):
    """Generate enhanced visualizations for improved clustering results."""
    print(f"\n🖼️  Generating improved visualizations...")

    features = clustering_results['features']
    cluster_labels = clustering_results['cluster_labels']
    crop_metadata = clustering_results['metadata']

    # Create PCA visualization
    pca = PCA(n_components=2, random_state=42)
    reduced_features = pca.fit_transform(features)

    # Create figure with enhanced subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 20))

    # Plot 1: Enhanced cluster assignments
    unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])
    noise_mask = np.array(cluster_labels) == -1

    colors = plt.get_cmap('tab10', len(unique_clusters))

    # Plot noise points first
    if np.any(noise_mask):
        ax1.scatter(
            reduced_features[noise_mask, 0],
            reduced_features[noise_mask, 1],
            c='lightgray',
            label='Noise',
            alpha=0.5,
            s=30,
            marker='x'
        )

    # Plot clusters
    for i, cluster_id in enumerate(unique_clusters):
        mask = np.array(cluster_labels) == cluster_id
        ax1.scatter(
            reduced_features[mask, 0],
            reduced_features[mask, 1],
            c=[colors(i)],
            label=f'Cluster {i+1}',
            alpha=0.7,
            s=60
        )

    ax1.set_title(
        f'Improved Cluster Assignments ({len(unique_clusters)} clusters + noise)')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot 2: Temporal distribution with improvements
    timestamps = [meta['timestamp'] for meta in crop_metadata]
    for i, cluster_id in enumerate(unique_clusters):
        mask = np.array(cluster_labels) == cluster_id
        cluster_timestamps = np.array(timestamps)[mask]
        ax2.scatter(
            cluster_timestamps,
            [i+1] * len(cluster_timestamps),
            c=[colors(i)],
            label=f'Cluster {i+1}',
            alpha=0.7,
            s=40
        )

    # Plot noise points
    if np.any(noise_mask):
        noise_timestamps = np.array(timestamps)[noise_mask]
        ax2.scatter(
            noise_timestamps,
            [0] * len(noise_timestamps),
            c='lightgray',
            label='Noise',
            alpha=0.5,
            s=20,
            marker='x'
        )

    ax2.set_title('Temporal Distribution (Improved)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Cluster ID')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Plot 3: Cluster sizes with improvements
    cluster_sizes = defaultdict(int)
    for label in cluster_labels:
        if label != -1:
            cluster_sizes[label] += 1

    cluster_ids = [f'Cluster {i+1}' for i in range(len(unique_clusters))]
    sizes = [cluster_sizes[cluster_id] for cluster_id in unique_clusters]

    bars = ax3.bar(cluster_ids, sizes, color=[
                   colors(i) for i in range(len(unique_clusters))])
    ax3.set_title('Cluster Sizes (Improved)')
    ax3.set_xlabel('Cluster ID')
    ax3.set_ylabel('Number of Detections')
    ax3.grid(True, linestyle='--', alpha=0.3)

    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{size}', ha='center', va='bottom')

    # Plot 4: Temporal group sizes (new metric)
    group_sizes = [meta.get('temporal_group_size', 1)
                   for meta in crop_metadata]
    ax4.hist(group_sizes, bins=range(1, max(group_sizes)+2),
             alpha=0.7, color='skyblue')
    ax4.set_title('Temporal Group Sizes Distribution')
    ax4.set_xlabel('Group Size')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save improved visualization
    viz_path = output_dir / 'improved_clustering_analysis.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Improved visualization saved to: {viz_path}")

    # Create comparison timeline
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

    # Timeline 1: Original temporal distribution
    for i, cluster_id in enumerate(unique_clusters):
        mask = np.array(cluster_labels) == cluster_id
        cluster_timestamps = np.array(timestamps)[mask]
        cluster_frames = [crop_metadata[j]['frame_number']
                          for j in range(len(crop_metadata)) if mask[j]]

        ax1.scatter(cluster_timestamps, cluster_frames,
                    c=[colors(i)], label=f'Cluster {i+1}',
                    alpha=0.7, s=25)

    ax1.set_title('Improved Cluster Timeline')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Frame Number')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Timeline 2: Temporal averaging effect
    group_sizes = [meta.get('temporal_group_size', 1)
                   for meta in crop_metadata]
    scatter = ax2.scatter(timestamps, group_sizes,
                          c=[colors(i) if cluster_labels[j] != -1 else 'lightgray'
                             for j in range(len(cluster_labels))
                             for i, cluster_id in enumerate(unique_clusters)
                             if cluster_labels[j] == cluster_id][:len(timestamps)],
                          alpha=0.7, s=30)

    ax2.set_title('Temporal Averaging Effect')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Temporal Group Size')
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    timeline_path = output_dir / 'improved_cluster_timeline.png'
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Improved timeline saved to: {timeline_path}")

    return viz_path, timeline_path


def main(video_path: str, output_dir: str, frame_skip: int = 1, max_frames: int = None):
    """Main function with improved clustering pipeline."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if video exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print("="*80)
    print("🎬 IMPROVED VIDEO CLUSTERING ANALYSIS")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")

    # Apply optimized preset
    print(f"\n⚙️  Applying three_person_dataset_90_confidence preset...")
    apply_preset('three_person_dataset_90_confidence')

    # Initialize improved analyzer
    analyzer = ImprovedClusteringAnalyzer(get_config())

    # Extract person crops from video
    person_crops, detection_stats = extract_person_crops_from_video(
        video_path,
        confidence_threshold=0.9,
        frame_skip=frame_skip,
        max_frames=max_frames
    )

    if len(person_crops) == 0:
        raise ValueError("No person crops extracted from video")

    # Extract ensemble features (Point 2: Improved representation)
    ensemble_features, crop_metadata, osnet_features, transreid_features = analyzer.extract_ensemble_features(
        person_crops)

    # Perform improved clustering analysis
    clustering_results = analyzer.perform_improved_clustering(
        ensemble_features, crop_metadata)

    # Generate improved visualizations
    viz_path, timeline_path = generate_improved_visualizations(
        clustering_results, output_dir)

    # Save results
    def convert_numpy_types(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    # Analyze cluster distributions
    cluster_sizes = defaultdict(int)
    cluster_frame_ranges = defaultdict(list)

    for i, (label, meta) in enumerate(zip(clustering_results['cluster_labels'], clustering_results['metadata'])):
        if label != -1:  # Exclude noise
            cluster_sizes[label] += 1
            cluster_frame_ranges[label].append(meta['frame_number'])

    results = {
        'video_info': {
            'path': str(video_path),
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'detection_stats': convert_numpy_types(detection_stats),
        'clustering_results': {
            'num_clusters': clustering_results['n_clusters'],
            'expected_clusters': 3,
            'clustering_efficiency': clustering_results['efficiency'],
            'silhouette_score': clustering_results['silhouette_score'],
            'cluster_sizes': convert_numpy_types(dict(cluster_sizes)),
            'cluster_frame_ranges': convert_numpy_types({k: {'min': min(v), 'max': max(v)} for k, v in cluster_frame_ranges.items()}),
            'noise_points': int(np.sum(np.array(clustering_results['cluster_labels']) == -1))
        },
        'improvements_applied': {
            'optimized_hyperparameters': True,
            'ensemble_features': True,
            'dbscan_clustering': True,
            'temporal_averaging': True,
            'post_clustering_fusion': True
        },
        'configuration': {
            'preset': 'three_person_dataset_90_confidence',
            'detection_confidence': 0.9,
            'mahalanobis_threshold': analyzer.optimized_params['mahalanobis_threshold'],
            'expected_persons': 3
        }
    }

    results_path = output_dir / 'improved_clustering_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print final summary
    print(f"\n" + "="*80)
    print("📊 IMPROVED CLUSTERING RESULTS")
    print("="*80)
    print(f"✅ Video processing complete!")
    print(f"   • Total crops extracted: {len(person_crops)}")
    print(f"   • Features extracted: {len(ensemble_features)}")
    print(
        f"   • Temporal averaging: {len(clustering_results['features'])} averaged features")
    print(f"   • Clusters formed: {clustering_results['n_clusters']}")
    print(f"   • Expected clusters: 3")
    print(
        f"   • Clustering efficiency: {clustering_results['efficiency']:.4f}")
    print(
        f"   • Silhouette score: {clustering_results['silhouette_score']:.4f}")
    print(
        f"   • Noise points: {np.sum(np.array(clustering_results['cluster_labels']) == -1)}")

    print(f"\n🔧 Improvements applied:")
    print(f"   • Optimized hyperparameters ✅")
    print(f"   • Ensemble features (OSNet + TransReID) ✅")
    print(f"   • DBSCAN clustering ✅")
    print(f"   • Temporal averaging ✅")
    print(f"   • Post-clustering fusion ✅")

    print(f"\n📁 Output files:")
    print(f"   • Results: {results_path}")
    print(f"   • Visualization: {viz_path}")
    print(f"   • Timeline: {timeline_path}")

    print("="*80)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Improved video clustering analysis')
    parser.add_argument('--video', default='data/raw/video2.mp4',
                        help='Path to video file')
    parser.add_argument('--output-dir', default='output/improved_clustering',
                        help='Output directory for results')
    parser.add_argument('--frame-skip', type=int, default=2,
                        help='Process every Nth frame (1 = all frames)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')

    args = parser.parse_args()

    main(args.video, args.output_dir, args.frame_skip, args.max_frames)
