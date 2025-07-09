#!/usr/bin/env python3
"""
Optimized Video Clustering Analysis Script
==========================================
This script implements optimized clustering improvements:
1. Efficient model loading and feature extraction
2. Improved hyperparameters and post-clustering fusion
3. DBSCAN clustering with better parameter tuning
4. Selective temporal processing
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


class OptimizedClusteringAnalyzer:
    """Optimized clustering analyzer with improved performance."""

    def __init__(self, config):
        self.config = config
        self.reid_cfg = get_config('reid')
        self.clustering_cfg = get_config('gaussian_clustering')

        # Initialize single instance of extractors
        print("🔧 Initializing ReID extractors...")
        self.osnet_extractor = OSNetReID()

        # Optimized parameters (Point 1: Better hyperparameters)
        self.optimized_params = {
            'mahalanobis_threshold': 6.0,    # Reduced from 7.0
            'min_samples_per_cluster': 4,    # Balanced
            'fusion_threshold': 4.8,         # 6.0 * 0.8
            'post_fusion_threshold': 0.3,    # For cosine distance
            'temporal_window': 2.0,          # 2 seconds window
            'dbscan_eps': 0.15,              # Conservative eps
            'dbscan_min_samples': 3,         # Minimum samples for core points
        }

        print(f"   ✅ Optimized parameters loaded:")
        print(
            f"     • Mahalanobis threshold: {self.optimized_params['mahalanobis_threshold']}")
        print(f"     • DBSCAN eps: {self.optimized_params['dbscan_eps']}")
        print(
            f"     • Temporal window: {self.optimized_params['temporal_window']}s")

    def extract_features_efficiently(self, person_crops):
        """Extract features efficiently without reloading model (Point 2)."""
        print(f"\n🚀 Extracting features from {len(person_crops)} crops...")

        features_list = []
        crop_metadata = []

        for idx, crop_data in enumerate(tqdm(person_crops, desc="Extracting features")):
            crop = crop_data['crop']

            try:
                # Extract features using single model instance
                features = self.osnet_extractor.extract_features(crop)

                if features is not None and len(features) > 0:
                    features_list.append(features[0])
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

        if len(features_list) == 0:
            raise ValueError("No features could be extracted from any crops")

        # Normalize features for better clustering
        features_array = np.vstack(features_list)
        normalized_features = normalize(features_array, norm='l2')

        print(f"   ✅ Extracted {len(normalized_features)} feature vectors")
        print(f"   • Feature dimension: {normalized_features.shape[1]}D")

        return normalized_features, crop_metadata

    def apply_selective_temporal_processing(self, features, crop_metadata):
        """Apply selective temporal processing (Point 4)."""
        print(f"\n🕐 Applying selective temporal processing...")

        temporal_window = self.optimized_params['temporal_window']

        # Group by temporal windows but keep individual samples
        time_groups = defaultdict(list)
        for idx, meta in enumerate(crop_metadata):
            time_key = int(meta['timestamp'] / temporal_window)
            time_groups[time_key].append({
                'feature': features[idx],
                'metadata': meta,
                'original_idx': idx
            })

        # Apply light temporal smoothing (average only if many samples in window)
        processed_features = []
        processed_metadata = []

        for time_key, group in time_groups.items():
            if len(group) > 5:  # Only average if many samples
                # Take top 3 highest confidence samples and average
                sorted_group = sorted(
                    group, key=lambda x: x['metadata']['confidence'], reverse=True)
                top_samples = sorted_group[:3]

                avg_feature = np.mean([item['feature']
                                      for item in top_samples], axis=0)
                best_meta = top_samples[0]['metadata'].copy()
                best_meta['temporal_group_size'] = len(group)

                processed_features.append(avg_feature)
                processed_metadata.append(best_meta)
            else:
                # Keep individual samples
                for item in group:
                    item['metadata']['temporal_group_size'] = len(group)
                    processed_features.append(item['feature'])
                    processed_metadata.append(item['metadata'])

        processed_features = np.vstack(processed_features)

        print(f"   ✅ Temporal processing complete:")
        print(f"   • Original features: {len(features)}")
        print(f"   • Processed features: {len(processed_features)}")
        print(
            f"   • Compression ratio: {len(processed_features)/len(features):.2f}")

        return processed_features, processed_metadata

    def perform_adaptive_dbscan(self, features, crop_metadata):
        """Perform DBSCAN with adaptive parameters (Point 3)."""
        print(f"\n🔄 Performing adaptive DBSCAN clustering...")

        # Start with conservative parameters
        base_eps = self.optimized_params['dbscan_eps']
        min_samples = self.optimized_params['dbscan_min_samples']

        # Try different eps values to find optimal clustering
        eps_values = [base_eps * 0.8, base_eps, base_eps * 1.2, base_eps * 1.5]
        best_result = None
        best_score = -1

        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = dbscan.fit_predict(features)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # Score based on cluster count and noise ratio
            if n_clusters > 0:
                noise_ratio = n_noise / len(labels)
                cluster_diff = abs(n_clusters - 3)
                # Prefer 3 clusters, avoid division by zero
                cluster_score = 1.0 / (cluster_diff + 0.1)
                # Penalize too much noise
                noise_penalty = max(0, 1 - noise_ratio * 2)
                total_score = cluster_score * noise_penalty

                if total_score > best_score:
                    best_score = total_score
                    best_result = {
                        'labels': labels,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'eps': eps,
                        'score': total_score
                    }

        if best_result is None:
            # Fallback to simple clustering
            dbscan = DBSCAN(
                eps=base_eps, min_samples=min_samples, metric='cosine')
            labels = dbscan.fit_predict(features)
            best_result = {
                'labels': labels,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'n_noise': list(labels).count(-1),
                'eps': base_eps,
                'score': 0.0
            }

        print(f"   ✅ DBSCAN clustering complete:")
        print(f"   • Clusters formed: {best_result['n_clusters']}")
        print(f"   • Noise points: {best_result['n_noise']}")
        print(f"   • Optimal eps: {best_result['eps']:.4f}")
        print(f"   • Clustering score: {best_result['score']:.4f}")

        return best_result['labels'], best_result['n_clusters']

    def perform_intelligent_fusion(self, features, cluster_labels, crop_metadata):
        """Perform intelligent post-clustering fusion (Point 1)."""
        print(f"\n🔗 Performing intelligent cluster fusion...")

        unique_clusters = [c for c in set(cluster_labels) if c != -1]

        if len(unique_clusters) <= 3:
            print("   ✅ No fusion needed - optimal cluster count achieved")
            return cluster_labels

        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id in unique_clusters:
            cluster_mask = np.array(cluster_labels) == cluster_id
            cluster_features = features[cluster_mask]
            cluster_times = [crop_metadata[i]['timestamp'] for i in range(
                len(cluster_labels)) if cluster_labels[i] == cluster_id]

            cluster_stats[cluster_id] = {
                'centroid': np.mean(cluster_features, axis=0),
                'size': len(cluster_features),
                'time_span': (min(cluster_times), max(cluster_times)),
                'avg_confidence': np.mean([crop_metadata[i]['confidence'] for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_id])
            }

        # Find clusters to merge based on similarity and size
        fusion_mapping = {}
        merged_clusters = set()

        # Sort clusters by size (merge smaller into larger)
        sorted_clusters = sorted(
            unique_clusters, key=lambda x: cluster_stats[x]['size'], reverse=True)

        for i, cluster_i in enumerate(sorted_clusters):
            if cluster_i in merged_clusters:
                continue

            for j, cluster_j in enumerate(sorted_clusters[i+1:], i+1):
                if cluster_j in merged_clusters:
                    continue

                # Calculate similarity
                similarity = np.dot(
                    cluster_stats[cluster_i]['centroid'], cluster_stats[cluster_j]['centroid'])

                # Check temporal overlap
                time_i = cluster_stats[cluster_i]['time_span']
                time_j = cluster_stats[cluster_j]['time_span']
                overlap = max(
                    0, min(time_i[1], time_j[1]) - max(time_i[0], time_j[0]))
                total_span = max(time_i[1], time_j[1]) - \
                    min(time_i[0], time_j[0])
                temporal_overlap = overlap / total_span if total_span > 0 else 0

                # Merge if similar and reasonable temporal relationship
                if similarity > 0.85 and (temporal_overlap < 0.3 or cluster_stats[cluster_j]['size'] < 10):
                    fusion_mapping[cluster_j] = cluster_i
                    merged_clusters.add(cluster_j)
                    print(
                        f"   • Merging cluster {cluster_j} ({cluster_stats[cluster_j]['size']} samples) into {cluster_i} (similarity: {similarity:.4f})")

        # Apply fusion mapping
        fused_labels = cluster_labels.copy()
        for i, label in enumerate(cluster_labels):
            if label in fusion_mapping:
                fused_labels[i] = fusion_mapping[label]

        n_clusters_after = len(set(fused_labels)) - \
            (1 if -1 in fused_labels else 0)
        print(f"   ✅ Cluster fusion complete:")
        print(f"   • Clusters before: {len(unique_clusters)}")
        print(f"   • Clusters after: {n_clusters_after}")

        return fused_labels

    def perform_optimized_clustering(self, features, crop_metadata):
        """Perform optimized clustering with all improvements."""
        print(f"\n🎯 Performing optimized clustering analysis...")

        # Step 1: Apply selective temporal processing
        processed_features, processed_metadata = self.apply_selective_temporal_processing(
            features, crop_metadata)

        # Step 2: Perform adaptive DBSCAN
        cluster_labels, n_clusters = self.perform_adaptive_dbscan(
            processed_features, processed_metadata)

        # Step 3: Apply intelligent fusion
        final_labels = self.perform_intelligent_fusion(
            processed_features, cluster_labels, processed_metadata)

        # Calculate final metrics
        final_n_clusters = len(set(final_labels)) - \
            (1 if -1 in final_labels else 0)

        # Calculate silhouette score
        if final_n_clusters > 1:
            non_noise_mask = np.array(final_labels) != -1
            if np.sum(non_noise_mask) > 1:
                silhouette = float(silhouette_score(
                    processed_features[non_noise_mask], np.array(final_labels)[non_noise_mask]))
            else:
                silhouette = -1.0
        else:
            silhouette = -1.0

        print(f"\n📊 Final Clustering Results:")
        print(f"   • Final clusters: {final_n_clusters}")
        print(f"   • Expected clusters: 3")
        print(f"   • Efficiency: {3/final_n_clusters:.4f}" if final_n_clusters >
              0 else "   • Efficiency: 0.0000")
        print(f"   • Silhouette score: {silhouette:.4f}")

        return {
            'features': processed_features,
            'metadata': processed_metadata,
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


def generate_optimized_visualizations(clustering_results, output_dir):
    """Generate optimized visualizations for clustering results."""
    print(f"\n🖼️  Generating optimized visualizations...")

    features = clustering_results['features']
    cluster_labels = clustering_results['cluster_labels']
    crop_metadata = clustering_results['metadata']

    # Create PCA visualization
    pca = PCA(n_components=2, random_state=42)
    reduced_features = pca.fit_transform(features)

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Plot 1: Cluster assignments
    unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])
    noise_mask = np.array(cluster_labels) == -1

    colors = plt.get_cmap('tab10', len(unique_clusters))

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

    # Plot noise points
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

    ax1.set_title(
        f'Optimized Cluster Assignments ({len(unique_clusters)} clusters)')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot 2: Temporal distribution
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

    ax2.set_title('Temporal Distribution (Optimized)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Cluster ID')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Plot 3: Cluster sizes
    cluster_sizes = defaultdict(int)
    for label in cluster_labels:
        if label != -1:
            cluster_sizes[label] += 1

    cluster_ids = [f'Cluster {i+1}' for i in range(len(unique_clusters))]
    sizes = [cluster_sizes[cluster_id] for cluster_id in unique_clusters]

    bars = ax3.bar(cluster_ids, sizes, color=[
                   colors(i) for i in range(len(unique_clusters))])
    ax3.set_title('Cluster Sizes')
    ax3.set_xlabel('Cluster ID')
    ax3.set_ylabel('Number of Detections')
    ax3.grid(True, linestyle='--', alpha=0.3)

    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{size}', ha='center', va='bottom')

    # Plot 4: Confidence distribution
    confidences = [meta['confidence'] for meta in crop_metadata]
    for i, cluster_id in enumerate(unique_clusters):
        mask = np.array(cluster_labels) == cluster_id
        cluster_confidences = np.array(confidences)[mask]
        ax4.hist(cluster_confidences, bins=15, alpha=0.7,
                 label=f'Cluster {i+1}', color=colors(i))

    ax4.set_title('Detection Confidence Distribution')
    ax4.set_xlabel('Detection Confidence')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save visualization
    viz_path = output_dir / 'optimized_clustering_analysis.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Optimized visualization saved to: {viz_path}")

    # Create timeline visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for i, cluster_id in enumerate(unique_clusters):
        mask = np.array(cluster_labels) == cluster_id
        cluster_timestamps = np.array(timestamps)[mask]
        cluster_frames = [crop_metadata[j]['frame_number']
                          for j in range(len(crop_metadata)) if mask[j]]

        ax.scatter(cluster_timestamps, cluster_frames,
                   c=[colors(i)], label=f'Cluster {i+1}',
                   alpha=0.7, s=25)

    ax.set_title('Optimized Cluster Timeline')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frame Number')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    timeline_path = output_dir / 'optimized_cluster_timeline.png'
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Timeline saved to: {timeline_path}")

    return viz_path, timeline_path


def main(video_path: str, output_dir: str, frame_skip: int = 1, max_frames: int = None):
    """Main function with optimized clustering pipeline."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if video exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print("="*80)
    print("🎬 OPTIMIZED VIDEO CLUSTERING ANALYSIS")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")

    # Apply optimized preset
    print(f"\n⚙️  Applying three_person_dataset_90_confidence preset...")
    apply_preset('three_person_dataset_90_confidence')

    # Initialize optimized analyzer
    analyzer = OptimizedClusteringAnalyzer(get_config())

    # Extract person crops from video
    person_crops, detection_stats = extract_person_crops_from_video(
        video_path,
        confidence_threshold=0.9,
        frame_skip=frame_skip,
        max_frames=max_frames
    )

    if len(person_crops) == 0:
        raise ValueError("No person crops extracted from video")

    # Extract features efficiently
    features, crop_metadata = analyzer.extract_features_efficiently(
        person_crops)

    # Perform optimized clustering analysis
    clustering_results = analyzer.perform_optimized_clustering(
        features, crop_metadata)

    # Generate optimized visualizations
    viz_path, timeline_path = generate_optimized_visualizations(
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
        if label != -1:
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
        'optimizations_applied': {
            'efficient_feature_extraction': True,
            'optimized_hyperparameters': True,
            'adaptive_dbscan': True,
            'selective_temporal_processing': True,
            'intelligent_fusion': True
        },
        'configuration': {
            'preset': 'three_person_dataset_90_confidence',
            'detection_confidence': 0.9,
            'mahalanobis_threshold': analyzer.optimized_params['mahalanobis_threshold'],
            'dbscan_eps': analyzer.optimized_params['dbscan_eps'],
            'expected_persons': 3
        }
    }

    results_path = output_dir / 'optimized_clustering_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print final summary
    print(f"\n" + "="*80)
    print("📊 OPTIMIZED CLUSTERING RESULTS")
    print("="*80)
    print(f"✅ Video processing complete!")
    print(f"   • Total crops extracted: {len(person_crops)}")
    print(f"   • Features extracted: {len(features)}")
    print(f"   • Processed features: {len(clustering_results['features'])}")
    print(f"   • Clusters formed: {clustering_results['n_clusters']}")
    print(f"   • Expected clusters: 3")
    print(
        f"   • Clustering efficiency: {clustering_results['efficiency']:.4f}")
    print(
        f"   • Silhouette score: {clustering_results['silhouette_score']:.4f}")
    print(
        f"   • Noise points: {np.sum(np.array(clustering_results['cluster_labels']) == -1)}")

    print(f"\n🔧 Optimizations applied:")
    print(f"   • Efficient feature extraction ✅")
    print(f"   • Optimized hyperparameters ✅")
    print(f"   • Adaptive DBSCAN clustering ✅")
    print(f"   • Selective temporal processing ✅")
    print(f"   • Intelligent cluster fusion ✅")

    print(f"\n📁 Output files:")
    print(f"   • Results: {results_path}")
    print(f"   • Visualization: {viz_path}")
    print(f"   • Timeline: {timeline_path}")

    print("="*80)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimized video clustering analysis')
    parser.add_argument('--video', default='data/raw/video2.mp4',
                        help='Path to video file')
    parser.add_argument('--output-dir', default='output/optimized_clustering',
                        help='Output directory for results')
    parser.add_argument('--frame-skip', type=int, default=2,
                        help='Process every Nth frame (1 = all frames)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')

    args = parser.parse_args()

    main(args.video, args.output_dir, args.frame_skip, args.max_frames)
