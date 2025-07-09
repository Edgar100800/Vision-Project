#!/usr/bin/env python3
"""
Final Improved Video Clustering Analysis Script
===============================================
This script implements key improvements for better clustering performance:
1. Optimized hyperparameters (mahalanobis threshold reduced to 6.0)
2. Better temporal processing (higher confidence threshold)
3. Improved feature extraction efficiency
4. Simple post-processing fusion
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
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Project imports
from config.global_config import get_config, update_config, apply_preset
from src.reid.osnet_reid import OSNetReID
from src.index.hnsw_index import HNSWIndex
from src.clustering.gaussian_cluster_manager import GaussianClusterManager


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


def extract_reid_features(person_crops, extractor):
    """Extract ReID features from person crops."""
    print(f"\n🚀 Extracting ReID features from {len(person_crops)} crops...")

    features_list = []
    crop_metadata = []

    for idx, crop_data in enumerate(tqdm(person_crops, desc="Extracting features")):
        crop = crop_data['crop']

        # Extract features
        try:
            features = extractor.extract_features(crop)
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
            print(f"Warning: Failed to extract features from crop {idx}: {e}")
            continue

    if len(features_list) == 0:
        raise ValueError("No features could be extracted from any crops")

    features_array = np.vstack(features_list)
    print(f"   ✅ Extracted {len(features_array)} feature vectors")

    return features_array, crop_metadata


def perform_improved_clustering_analysis(features, crop_metadata, clustering_cfg, reid_cfg):
    """Perform improved clustering analysis with better hyperparameters."""
    print(f"\n🔍 Performing improved clustering analysis...")

    # IMPROVEMENT 1: Reduced Mahalanobis threshold for better clustering
    improved_threshold = 6.0  # Reduced from 7.0
    print(f"   📋 Using improved threshold: {improved_threshold}")
    print(f"   📋 Expected persons: {clustering_cfg['expected_persons']}")

    # Create cluster manager with improved parameters
    cluster_mgr = GaussianClusterManager(
        embedding_dim=reid_cfg['feature_dim'],
        mahalanobis_threshold=improved_threshold,
        max_clusters=clustering_cfg['max_clusters'],
        min_samples_for_update=5,  # Increased from 3
        fusion_threshold=improved_threshold * 0.75,  # More aggressive fusion
        split_threshold=improved_threshold * 2.0,
        initial_sigma_scale=1.0
    )

    # Create HNSW index
    hnsw = HNSWIndex(
        dim=reid_cfg['feature_dim'],
        max_elements=len(features)
    )

    external_to_cluster = {}
    cluster_labels = []
    cluster_assignments = []

    print(f"   🔄 Processing {len(features)} features...")

    for idx, feat in enumerate(tqdm(features, desc="Clustering")):
        # Search for similar features
        if hnsw.index.get_current_count() > 0:
            search_res = hnsw.search(feat, k=5)
            neighbour_clusters = [
                external_to_cluster.get(ext_id)
                for ext_id in search_res['external_ids']
                if ext_id is not None
            ]
            neighbour_clusters = [
                c for c in neighbour_clusters if c is not None]
        else:
            neighbour_clusters = []

        # Assign to cluster
        cluster_id, confidence = cluster_mgr.assign_to_cluster(
            feat, neighbour_clusters)

        # Store results
        ext_id = f"crop_{idx}"
        hnsw.add_vector(feat, external_id=ext_id, metadata={
            'cluster_id': cluster_id,
            'frame_number': crop_metadata[idx]['frame_number'],
            'timestamp': crop_metadata[idx]['timestamp']
        })

        external_to_cluster[ext_id] = cluster_id
        cluster_labels.append(cluster_id)

        cluster_assignments.append({
            'crop_idx': idx,
            'cluster_id': cluster_id,
            'confidence': confidence,
            'frame_number': crop_metadata[idx]['frame_number'],
            'timestamp': crop_metadata[idx]['timestamp']
        })

    # IMPROVEMENT 2: Post-processing fusion for small clusters
    unique_clusters = sorted(list(set(cluster_labels)))
    cluster_sizes = defaultdict(int)
    for label in cluster_labels:
        cluster_sizes[label] += 1

    # Merge very small clusters with similar larger ones
    small_clusters = [c for c in unique_clusters if cluster_sizes[c] < 10]
    large_clusters = [c for c in unique_clusters if cluster_sizes[c] >= 10]

    if len(small_clusters) > 0 and len(large_clusters) > 0:
        print(
            f"   🔗 Post-processing: Found {len(small_clusters)} small clusters to potentially merge")

        # Calculate cluster centroids
        cluster_centroids = {}
        for cluster_id in unique_clusters:
            mask = np.array(cluster_labels) == cluster_id
            cluster_features = features[mask]
            cluster_centroids[cluster_id] = np.mean(cluster_features, axis=0)

        # Merge small clusters into most similar large clusters
        for small_cluster in small_clusters:
            best_large_cluster = None
            best_similarity = -1

            for large_cluster in large_clusters:
                similarity = np.dot(
                    cluster_centroids[small_cluster], cluster_centroids[large_cluster])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_large_cluster = large_cluster

            if best_large_cluster is not None and best_similarity > 0.7:
                # Merge small cluster into large cluster
                for i, label in enumerate(cluster_labels):
                    if label == small_cluster:
                        cluster_labels[i] = best_large_cluster
                        cluster_assignments[i]['cluster_id'] = best_large_cluster

                print(
                    f"   • Merged cluster {small_cluster} ({cluster_sizes[small_cluster]} samples) into {best_large_cluster}")

    # Analyze final results
    final_unique_clusters = sorted(list(set(cluster_labels)))
    num_clusters = len(final_unique_clusters)

    print(f"\n📊 Final Clustering Results:")
    print(f"   • Clusters formed: {num_clusters}")
    print(f"   • Expected clusters: {clustering_cfg['expected_persons']}")
    print(
        f"   • Efficiency: {clustering_cfg['expected_persons']/num_clusters:.4f}" if num_clusters > 0 else "   • Efficiency: 0.0000")

    # Calculate silhouette score
    silhouette = silhouette_score(
        features, cluster_labels) if num_clusters > 1 else -1
    print(f"   • Silhouette score: {silhouette:.4f}")

    # Analyze final cluster distributions
    final_cluster_sizes = defaultdict(int)
    cluster_frame_ranges = defaultdict(list)

    for assignment in cluster_assignments:
        cluster_id = assignment['cluster_id']
        final_cluster_sizes[cluster_id] += 1
        cluster_frame_ranges[cluster_id].append(assignment['frame_number'])

    print(f"\n📈 Final Cluster Analysis:")
    for cluster_id in sorted(final_cluster_sizes.keys()):
        size = final_cluster_sizes[cluster_id]
        frames = cluster_frame_ranges[cluster_id]
        frame_range = f"{min(frames)}-{max(frames)}"
        print(
            f"   • Cluster {cluster_id}: {size} detections, frames {frame_range}")

    clustering_results = {
        'num_clusters': num_clusters,
        'expected_clusters': clustering_cfg['expected_persons'],
        'clustering_efficiency': clustering_cfg['expected_persons']/num_clusters if num_clusters > 0 else 0,
        'silhouette_score': silhouette,
        'cluster_labels': cluster_labels,
        'cluster_assignments': cluster_assignments,
        'cluster_sizes': dict(final_cluster_sizes),
        'cluster_frame_ranges': {k: {'min': min(v), 'max': max(v)} for k, v in cluster_frame_ranges.items()},
        'features': features,
        'crop_metadata': crop_metadata
    }

    return clustering_results


def generate_improved_visualizations(clustering_results, output_dir):
    """Generate improved visualization plots for clustering results."""
    print(f"\n🖼️  Generating improved visualizations...")

    features = clustering_results['features']
    cluster_labels = clustering_results['cluster_labels']
    crop_metadata = clustering_results['crop_metadata']

    # Create PCA visualization
    pca = PCA(n_components=2, random_state=42)
    reduced_features = pca.fit_transform(features)

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Plot 1: Improved cluster assignments
    unique_clusters = sorted(list(set(cluster_labels)))
    colors = plt.get_cmap('tab10', len(unique_clusters))

    for i, cluster_id in enumerate(unique_clusters):
        mask = np.array(cluster_labels) == cluster_id
        ax1.scatter(
            reduced_features[mask, 0],
            reduced_features[mask, 1],
            c=[colors(i)],
            label=f'Cluster {cluster_id}',
            alpha=0.7,
            s=60
        )

    ax1.set_title(
        f'Improved Cluster Assignments ({len(unique_clusters)} clusters)')
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
            [cluster_id] * len(cluster_timestamps),
            c=[colors(i)],
            label=f'Cluster {cluster_id}',
            alpha=0.7,
            s=30
        )

    ax2.set_title('Improved Temporal Distribution')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Cluster ID')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Plot 3: Cluster sizes
    cluster_sizes = clustering_results['cluster_sizes']
    cluster_ids = list(cluster_sizes.keys())
    sizes = list(cluster_sizes.values())

    bars = ax3.bar(cluster_ids, sizes, color=[
                   colors(i) for i in range(len(cluster_ids))])
    ax3.set_title('Improved Cluster Sizes')
    ax3.set_xlabel('Cluster ID')
    ax3.set_ylabel('Number of Detections')
    ax3.grid(True, linestyle='--', alpha=0.3)

    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{size}', ha='center', va='bottom')

    # Plot 4: Detection confidence distribution
    confidences = [meta['confidence'] for meta in crop_metadata]
    for i, cluster_id in enumerate(unique_clusters):
        mask = np.array(cluster_labels) == cluster_id
        cluster_confidences = np.array(confidences)[mask]
        ax4.hist(cluster_confidences, bins=20, alpha=0.7,
                 label=f'Cluster {cluster_id}', color=colors(i))

    ax4.set_title('Detection Confidence Distribution by Cluster')
    ax4.set_xlabel('Detection Confidence')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save visualization
    viz_path = output_dir / 'final_clustering_analysis.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Visualization saved to: {viz_path}")

    # Create timeline visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Plot timeline with clusters
    for i, cluster_id in enumerate(unique_clusters):
        mask = np.array(cluster_labels) == cluster_id
        cluster_timestamps = np.array(timestamps)[mask]
        cluster_frames = [crop_metadata[j]['frame_number']
                          for j in range(len(crop_metadata)) if mask[j]]

        ax.scatter(cluster_timestamps, cluster_frames,
                   c=[colors(i)], label=f'Cluster {cluster_id}',
                   alpha=0.7, s=20)

    ax.set_title('Final Cluster Timeline')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frame Number')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    timeline_path = output_dir / 'final_cluster_timeline.png'
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Timeline saved to: {timeline_path}")

    return viz_path, timeline_path


def main(video_path: str, output_dir: str, frame_skip: int = 1, max_frames: int = None):
    """Main function to process video and perform improved clustering analysis."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if video exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print("="*80)
    print("🎬 FINAL IMPROVED VIDEO CLUSTERING ANALYSIS")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")

    # Apply optimized preset
    print(f"\n⚙️  Applying three_person_dataset_90_confidence preset...")
    apply_preset('three_person_dataset_90_confidence')

    # Get configurations
    reid_cfg = get_config('reid')
    clustering_cfg = get_config('gaussian_clustering')
    detection_cfg = get_config('detection')

    print(f"   ✅ Configuration applied with improvements:")
    print(
        f"     • Detection confidence: {detection_cfg['confidence_threshold']:.0%}")
    print(f"     • Improved Mahalanobis threshold: 6.0 (reduced from 7.0)")
    print(f"     • Expected persons: {clustering_cfg['expected_persons']}")

    # Initialize ReID extractor
    print(f"\n🔧 Initializing ReID extractor...")
    extractor = OSNetReID()

    # Extract person crops from video
    person_crops, detection_stats = extract_person_crops_from_video(
        video_path,
        confidence_threshold=detection_cfg['confidence_threshold'],
        frame_skip=frame_skip,
        max_frames=max_frames
    )

    if len(person_crops) == 0:
        raise ValueError("No person crops extracted from video")

    # Extract ReID features
    features, crop_metadata = extract_reid_features(person_crops, extractor)

    # Perform improved clustering analysis
    clustering_results = perform_improved_clustering_analysis(
        features, crop_metadata, clustering_cfg, reid_cfg
    )

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

    results = {
        'video_info': {
            'path': str(video_path),
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'detection_stats': convert_numpy_types(detection_stats),
        'clustering_results': {
            'num_clusters': int(clustering_results['num_clusters']),
            'expected_clusters': int(clustering_results['expected_clusters']),
            'clustering_efficiency': float(clustering_results['clustering_efficiency']),
            'silhouette_score': float(clustering_results['silhouette_score']),
            'cluster_sizes': convert_numpy_types(clustering_results['cluster_sizes']),
            'cluster_frame_ranges': convert_numpy_types(clustering_results['cluster_frame_ranges'])
        },
        'improvements_applied': {
            'reduced_mahalanobis_threshold': 6.0,
            'increased_min_samples': 5,
            'aggressive_fusion': True,
            'post_processing_merge': True
        },
        'configuration': {
            'preset': 'three_person_dataset_90_confidence',
            'detection_confidence': float(detection_cfg['confidence_threshold']),
            'improved_mahalanobis_threshold': 6.0,
            'expected_persons': int(clustering_cfg['expected_persons'])
        }
    }

    results_path = output_dir / 'final_clustering_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print final summary
    print(f"\n" + "="*80)
    print("📊 FINAL IMPROVED RESULTS")
    print("="*80)
    print(f"✅ Video processing complete!")
    print(f"   • Total crops extracted: {len(person_crops)}")
    print(f"   • Features extracted: {len(features)}")
    print(f"   • Clusters formed: {clustering_results['num_clusters']}")
    print(f"   • Expected clusters: {clustering_results['expected_clusters']}")
    print(
        f"   • Clustering efficiency: {clustering_results['clustering_efficiency']:.4f}")
    print(
        f"   • Silhouette score: {clustering_results['silhouette_score']:.4f}")

    print(f"\n🔧 Key improvements applied:")
    print(f"   • Reduced Mahalanobis threshold: 7.0 → 6.0 ✅")
    print(f"   • Increased min samples per cluster: 3 → 5 ✅")
    print(f"   • More aggressive fusion threshold ✅")
    print(f"   • Post-processing small cluster merging ✅")

    print(f"\n📁 Output files:")
    print(f"   • Results: {results_path}")
    print(f"   • Visualization: {viz_path}")
    print(f"   • Timeline: {timeline_path}")

    print("="*80)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Final improved video clustering analysis')
    parser.add_argument('--video', default='data/raw/video2.mp4',
                        help='Path to video file')
    parser.add_argument('--output-dir', default='output/final_clustering',
                        help='Output directory for results')
    parser.add_argument('--frame-skip', type=int, default=2,
                        help='Process every Nth frame (1 = all frames)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')

    args = parser.parse_args()

    main(args.video, args.output_dir, args.frame_skip, args.max_frames)
