#!/usr/bin/env python3
"""
Cluster Evaluation Script with Person Detection Validation
==========================================================
This script analyzes person re-identification data with multiple sequences per person.
It first validates that images contain persons using YOLO detection before processing
with ReID features, then builds clusters and evaluates clustering performance.
"""

import os
import sys
import cv2
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Project imports
from config.global_config import get_config, update_config
from src.reid.osnet_reid import OSNetReID
from src.index.hnsw_index import HNSWIndex
from src.clustering.gaussian_cluster_manager import GaussianClusterManager


def load_images(data_dir: Path):
    """Iterate over all person folders and yield (image_path, person_id, sequence_id)."""
    image_paths, person_ids, sequence_ids = [], [], []

    # Sort folders to ensure consistent ordering
    person_dirs = sorted(data_dir.iterdir())

    for person_dir in person_dirs:
        if not person_dir.is_dir():
            continue

        # Parse person and sequence IDs from folder name (e.g., "person_A_1")
        parts = person_dir.name.split('_')
        if len(parts) != 3:
            continue

        person_id = parts[1]  # A, B, or C
        sequence_id = parts[2]  # 1 or 2

        for img_path in sorted(person_dir.glob('*.jpg')):
            image_paths.append(img_path)
            person_ids.append(person_id)
            sequence_ids.append(f"{person_id}_{sequence_id}")

    return image_paths, person_ids, sequence_ids


def preprocess_image_cv2(path: Path):
    """Read image in BGR format using OpenCV."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def validate_person_detection(image_paths, person_ids, sequence_ids, confidence_threshold=0.8):
    """
    Validate that images contain persons using YOLO detection.

    Args:
        image_paths: List of image paths
        person_ids: List of person IDs
        sequence_ids: List of sequence IDs
        confidence_threshold: Minimum confidence for person detection (0.0-1.0)

    Returns:
        Filtered lists containing only images with valid person detections
    """
    print(
        f"🔍 Validating person detection with confidence threshold: {confidence_threshold:.0%}")

    # Load YOLO model
    detection_cfg = get_config('detection')
    yolo_model = YOLO(detection_cfg['model_path'])

    valid_image_paths = []
    valid_person_ids = []
    valid_sequence_ids = []
    detection_stats = {
        'total_images': len(image_paths),
        'valid_detections': 0,
        'invalid_detections': 0,
        'confidence_threshold': confidence_threshold
    }

    print(f"   Processing {len(image_paths)} images...")

    for img_path, person_id, sequence_id in tqdm(zip(image_paths, person_ids, sequence_ids),
                                                 total=len(image_paths), desc="Validating"):
        img = preprocess_image_cv2(img_path)

        # Run YOLO detection
        results = yolo_model(img, verbose=False)

        # Check if any person detection meets confidence threshold
        valid_person_detected = False
        for result in results:
            if result.boxes is not None:
                # Filter for person class (class 0 in COCO)
                person_boxes = result.boxes[result.boxes.cls == 0]
                if len(person_boxes) > 0:
                    # Check if any person detection meets confidence threshold
                    max_confidence = person_boxes.conf.max().item()
                    if max_confidence >= confidence_threshold:
                        valid_person_detected = True
                        break

        if valid_person_detected:
            valid_image_paths.append(img_path)
            valid_person_ids.append(person_id)
            valid_sequence_ids.append(sequence_id)
            detection_stats['valid_detections'] += 1
        else:
            detection_stats['invalid_detections'] += 1

    # Calculate statistics
    validation_rate = detection_stats['valid_detections'] / \
        detection_stats['total_images']

    print(f"   ✅ Validation complete:")
    print(f"     - Total images: {detection_stats['total_images']}")
    print(f"     - Valid detections: {detection_stats['valid_detections']}")
    print(
        f"     - Invalid detections: {detection_stats['invalid_detections']}")
    print(f"     - Validation rate: {validation_rate:.2%}")

    return valid_image_paths, valid_person_ids, valid_sequence_ids, detection_stats


def extract_features(image_paths, person_ids, sequence_ids, extractor):
    """Extract features from all images."""
    features_list = []

    print("🚀 Extracting ReID features...")
    for idx, (img_path, person_id, sequence_id) in enumerate(tqdm(list(zip(
            image_paths, person_ids, sequence_ids)), total=len(image_paths))):

        img = preprocess_image_cv2(img_path)
        feat = extractor.extract_features(img)[0]
        features_list.append(feat)

    return np.vstack(features_list)


def test_clustering_threshold(features, person_ids, clustering_cfg, reid_cfg, target_clusters=3):
    """Test different mahalanobis thresholds to find optimal clustering."""
    print(
        f"\n🔍 Testing different thresholds to achieve {target_clusters} clusters...")

    # Test different thresholds
    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
    results = []

    for threshold in thresholds:
        print(f"   Testing threshold: {threshold}")

        # Create cluster manager with current threshold
        cluster_mgr = GaussianClusterManager(
            embedding_dim=reid_cfg['feature_dim'],
            mahalanobis_threshold=threshold,
            max_clusters=clustering_cfg['max_clusters'],
            min_samples_for_update=clustering_cfg['min_samples_per_cluster'],
            fusion_threshold=threshold * 0.8,
            split_threshold=threshold * 2.0,
            initial_sigma_scale=1.0
        )

        # Create HNSW index
        hnsw = HNSWIndex(
            dim=reid_cfg['feature_dim'],
            max_elements=len(features)
        )

        external_to_cluster = {}
        cluster_labels = []

        # Process features
        for idx, feat in enumerate(features):
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

            cluster_id, _ = cluster_mgr.assign_to_cluster(
                feat, neighbour_clusters)

            ext_id = f"img_{idx}"
            hnsw.add_vector(feat, external_id=ext_id, metadata={
                            'cluster_id': cluster_id})

            external_to_cluster[ext_id] = cluster_id
            cluster_labels.append(cluster_id)

        # Evaluate results
        num_clusters = len(set(cluster_labels))
        ari = adjusted_rand_score(person_ids, cluster_labels)
        sil = silhouette_score(
            features, cluster_labels) if num_clusters > 1 else -1

        # Calculate efficiency (how close to target number of clusters)
        efficiency = target_clusters / num_clusters if num_clusters > 0 else 0

        results.append({
            'threshold': threshold,
            'num_clusters': num_clusters,
            'ari': ari,
            'silhouette': sil,
            'efficiency': efficiency,
            'cluster_labels': cluster_labels.copy()
        })

        print(
            f"     -> Clusters: {num_clusters}, ARI: {ari:.4f}, Efficiency: {efficiency:.4f}")

    # Find best threshold (prioritize efficiency, then ARI)
    best_result = max(results, key=lambda x: (x['efficiency'], x['ari']))

    print(f"\n🎯 Best threshold: {best_result['threshold']}")
    print(f"   -> Clusters: {best_result['num_clusters']}")
    print(f"   -> ARI: {best_result['ari']:.4f}")
    print(f"   -> Efficiency: {best_result['efficiency']:.4f}")

    return best_result


def evaluate_confidence_thresholds(image_paths, person_ids, sequence_ids, extractor, clustering_cfg, reid_cfg):
    """
    Evaluate clustering performance with different YOLO confidence thresholds.

    Args:
        image_paths: List of image paths
        person_ids: List of person IDs
        sequence_ids: List of sequence IDs
        extractor: ReID feature extractor
        clustering_cfg: Clustering configuration
        reid_cfg: ReID configuration

    Returns:
        Dictionary with results for each confidence threshold
    """
    confidence_thresholds = [0.80, 0.85, 0.90, 0.95]
    results = {}

    print(
        f"\n🧪 Evaluating {len(confidence_thresholds)} confidence thresholds...")

    for conf_threshold in confidence_thresholds:
        print(f"\n" + "="*60)
        print(f"🎯 Testing confidence threshold: {conf_threshold:.0%}")
        print("="*60)

        # Validate person detection with current threshold
        valid_paths, valid_person_ids, valid_sequence_ids, detection_stats = validate_person_detection(
            image_paths, person_ids, sequence_ids, conf_threshold
        )

        if len(valid_paths) == 0:
            print(
                f"❌ No valid images found with confidence threshold {conf_threshold:.0%}")
            results[conf_threshold] = {
                'detection_stats': detection_stats,
                'clustering_results': None,
                'error': 'No valid images'
            }
            continue

        # Extract features from valid images
        features = extract_features(
            valid_paths, valid_person_ids, valid_sequence_ids, extractor)

        # Test clustering thresholds
        best_clustering = test_clustering_threshold(
            features, valid_person_ids, clustering_cfg, reid_cfg)

        # Run final clustering with optimal threshold
        print(
            f"\n🎯 Final clustering with threshold: {best_clustering['threshold']}")

        cluster_mgr = GaussianClusterManager(
            embedding_dim=reid_cfg['feature_dim'],
            mahalanobis_threshold=best_clustering['threshold'],
            max_clusters=clustering_cfg['max_clusters'],
            min_samples_for_update=clustering_cfg['min_samples_per_cluster'],
            fusion_threshold=best_clustering['threshold'] * 0.8,
            split_threshold=best_clustering['threshold'] * 2.0,
            initial_sigma_scale=1.0
        )

        hnsw = HNSWIndex(
            dim=reid_cfg['feature_dim'],
            max_elements=len(features)
        )

        external_to_cluster = {}
        cluster_labels = []

        # Process features
        for idx, feat in enumerate(features):
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

            cluster_id, _ = cluster_mgr.assign_to_cluster(
                feat, neighbour_clusters)

            ext_id = f"img_{idx}"
            hnsw.add_vector(feat, external_id=ext_id, metadata={
                            'cluster_id': cluster_id})

            external_to_cluster[ext_id] = cluster_id
            cluster_labels.append(cluster_id)

        # Evaluate final clustering
        ari = adjusted_rand_score(valid_person_ids, cluster_labels)
        sil = silhouette_score(features, cluster_labels) if len(
            set(cluster_labels)) > 1 else -1

        # Per-person analysis
        person_cluster_analysis = {}
        for person in ['A', 'B', 'C']:
            person_indices = [i for i, p in enumerate(
                valid_person_ids) if p == person]
            if len(person_indices) > 0:
                person_clusters = [cluster_labels[i] for i in person_indices]
                unique_clusters = set(person_clusters)

                main_cluster = max(unique_clusters, key=person_clusters.count)
                purity = person_clusters.count(
                    main_cluster) / len(person_clusters)

                person_cluster_analysis[person] = {
                    'total_images': len(person_indices),
                    'clusters_assigned': list(unique_clusters),
                    'num_clusters': len(unique_clusters),
                    'main_cluster': main_cluster,
                    'purity': purity
                }
            else:
                person_cluster_analysis[person] = {
                    'total_images': 0,
                    'clusters_assigned': [],
                    'num_clusters': 0,
                    'main_cluster': None,
                    'purity': 0.0
                }

        # Store results
        clustering_results = {
            'adjusted_rand_score': ari,
            'silhouette_score': sil,
            'num_clusters_formed': len(set(cluster_labels)),
            'expected_clusters': clustering_cfg['expected_persons'],
            'clustering_efficiency': clustering_cfg['expected_persons'] / len(set(cluster_labels)) if len(set(cluster_labels)) > 0 else 0,
            'optimal_threshold': best_clustering['threshold'],
            'person_analysis': person_cluster_analysis,
            'features': features,
            'cluster_labels': cluster_labels,
            'valid_person_ids': valid_person_ids
        }

        results[conf_threshold] = {
            'detection_stats': detection_stats,
            'clustering_results': clustering_results,
            'error': None
        }

        # Print summary
        print(f"\n📊 Results for confidence {conf_threshold:.0%}:")
        print(f"   • Valid images: {detection_stats['valid_detections']}")
        print(f"   • Clusters formed: {len(set(cluster_labels))}")
        print(f"   • ARI: {ari:.4f}")
        print(
            f"   • Efficiency: {clustering_results['clustering_efficiency']:.4f}")

    return results


def main(data_dir: str, output_dir: str, max_images: int = None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"🔍 Loading dataset from {data_dir} ...")
    image_paths, person_ids, sequence_ids = load_images(data_dir)
    if max_images:
        image_paths = image_paths[:max_images]
        person_ids = person_ids[:max_images]
        sequence_ids = sequence_ids[:max_images]

    num_images = len(image_paths)
    num_persons = len(set(person_ids))
    num_sequences = len(set(sequence_ids))
    print(f"   ➜ {num_images} images")
    print(f"   ➜ {num_persons} unique persons")
    print(f"   ➜ {num_sequences} sequences")

    # Initialize models using optimized config for 3-person dataset
    from config.global_config import apply_preset

    # Apply three_person_dataset preset for optimal clustering
    apply_preset('three_person_dataset')

    reid_cfg = get_config('reid')
    indexing_cfg = get_config('indexing')
    clustering_cfg = get_config('gaussian_clustering')

    print(f"   ➜ Using clustering config:")
    print(
        f"     - Mahalanobis threshold: {clustering_cfg['mahalanobis_threshold']}")
    print(f"     - Expected persons: {clustering_cfg['expected_persons']}")
    print(f"     - Max clusters: {clustering_cfg['max_clusters']}")

    extractor = OSNetReID()

    # Evaluate different confidence thresholds
    all_results = evaluate_confidence_thresholds(
        image_paths, person_ids, sequence_ids, extractor, clustering_cfg, reid_cfg
    )

    # Save comprehensive results
    results_path = output_dir / 'confidence_threshold_analysis.json'

    # Prepare results for JSON serialization
    json_results = {}
    for conf_threshold, result in all_results.items():
        json_result = {
            'detection_stats': result['detection_stats'],
            'error': result['error']
        }

        if result['clustering_results'] is not None:
            clustering_results = result['clustering_results'].copy()
            # Remove non-serializable items
            clustering_results.pop('features', None)
            clustering_results.pop('cluster_labels', None)
            clustering_results.pop('valid_person_ids', None)
            json_result['clustering_results'] = clustering_results

        json_results[f"{conf_threshold:.0%}"] = json_result

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    # Generate visualizations for best performing threshold
    print(f"\n🖼️  Generating visualizations...")

    # Find best threshold based on clustering efficiency and ARI
    best_threshold = None
    best_score = -1

    for conf_threshold, result in all_results.items():
        if result['clustering_results'] is not None:
            clustering_results = result['clustering_results']
            # Combined score: efficiency + ARI
            score = clustering_results['clustering_efficiency'] + \
                clustering_results['adjusted_rand_score']
            if score > best_score:
                best_score = score
                best_threshold = conf_threshold

    if best_threshold is not None:
        print(f"   Using best threshold: {best_threshold:.0%}")
        best_results = all_results[best_threshold]['clustering_results']

        # Generate PCA visualization
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(best_results['features'])

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Ground truth plot
        unique_persons = sorted(list(set(best_results['valid_person_ids'])))
        person_colors = plt.get_cmap('tab10', len(unique_persons))
        for i, person in enumerate(unique_persons):
            mask = np.array(best_results['valid_person_ids']) == person
            ax1.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=[person_colors(i)],
                label=f"Person {person}",
                alpha=0.7,
                s=50
            )
        ax1.set_title(f'Ground Truth (Confidence: {best_threshold:.0%})')
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Second Principal Component')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Cluster assignment plot
        unique_clusters = sorted(list(set(best_results['cluster_labels'])))
        cluster_colors = plt.get_cmap('tab20', len(unique_clusters))
        for i, cluster in enumerate(unique_clusters):
            mask = np.array(best_results['cluster_labels']) == cluster
            ax2.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=[cluster_colors(i)],
                label=f"Cluster {i+1}",
                alpha=0.7,
                s=50
            )
        ax2.set_title(f'Cluster Assignment ({len(unique_clusters)} clusters)')
        ax2.set_xlabel('First Principal Component')
        ax2.set_ylabel('Second Principal Component')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'clustering_with_detection_validation.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Print final summary
    print(f"\n" + "="*80)
    print("📊 FINAL SUMMARY - CONFIDENCE THRESHOLD ANALYSIS")
    print("="*80)

    for conf_threshold, result in all_results.items():
        detection_stats = result['detection_stats']
        print(f"\n🎯 Confidence {conf_threshold:.0%}:")
        print(
            f"   • Valid images: {detection_stats['valid_detections']}/{detection_stats['total_images']} ({detection_stats['valid_detections']/detection_stats['total_images']:.1%})")

        if result['clustering_results'] is not None:
            clustering = result['clustering_results']
            print(f"   • Clusters formed: {clustering['num_clusters_formed']}")
            print(f"   • ARI: {clustering['adjusted_rand_score']:.4f}")
            print(
                f"   • Efficiency: {clustering['clustering_efficiency']:.4f}")

            # Show best person purity
            best_purity = max([p['purity'] for p in clustering['person_analysis'].values(
            ) if p['total_images'] > 0], default=0)
            print(f"   • Best person purity: {best_purity:.4f}")
        else:
            print(f"   • ❌ {result['error']}")

    print(f"\n✅ Results saved to: {output_dir}")
    print(f"   • Analysis: {results_path}")
    print(
        f"   • Visualization: {output_dir / 'clustering_with_detection_validation.png'}")
    print("="*80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate clustering with person detection validation')
    parser.add_argument('--data-dir', default='data/re-id',
                        help='Directory containing person folders')
    parser.add_argument('--output-dir', default='output/visualizations/test_cluster',
                        help='Output directory for results')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Cap on number of images to process')

    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.max_images)
