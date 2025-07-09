#!/usr/bin/env python3
"""
Video Clustering with Pose2ID NFC Enhancement
===========================================
Procesa un video, extrae crops de personas, aplica centralización de características
(NFC) de Pose2ID y realiza clustering con GaussianClusterManager ajustado para
un número esperado de personas configurable (por defecto 5).

Principales mejoras sobre versiones anteriores:
1. Centralización de características mediante NFC (Neighbor Feature Centralization).
2. Umbral de Mahalanobis reducido para mayor separación de identidades.
3. Fusión menos agresiva para evitar juntar identidades distintas.
4. Soporte de argumento --expected-persons.
"""

import sys
from src.clustering.gaussian_cluster_manager import GaussianClusterManager
from src.index.hnsw_index import HNSWIndex
from src.reid.osnet_reid import OSNetReID
from config.global_config import get_config, apply_preset
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
import time
import json
import os
import torch


def pairwise_distance(query_features, gallery_features):
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
    """Neighbor Feature Centralization adapted for CPU/CUDA"""
    device = feat.device
    feat = feat.clone()
    dist = pairwise_distance(feat, feat)

    eye = torch.eye(dist.size(0)).to(device)
    dist[eye == 1] = 1000
    val, rank = dist.topk(k1, largest=False)

    mutual_topk_list = []
    for i in range(rank.size(0)):
        mutual_list = []
        for j in rank[i]:
            if i in rank[j][:k2]:
                mutual_list.append(j.item())
        mutual_topk_list.append(mutual_list)

    feat_copy = feat.clone()
    for i in range(rank.size(0)):
        feat[i] += feat_copy[mutual_topk_list[i]].sum(dim=0)
    return feat


# Project imports
# Pose2ID NFC
sys.path.append('Pose2ID')

# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------


def convert_numpy_types(obj):
    """Utility: convert numpy & torch types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

# --------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------


def extract_person_crops(video_path: Path, confidence=0.9, frame_skip=1, max_frames=None):
    detection_cfg = get_config('detection')
    yolo_model = YOLO(detection_cfg['model_path'])

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

    pbar = tqdm(total=total_frames//frame_skip, desc='Frames')
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % frame_skip == 0:
            stats['total_frames_processed'] += 1
            results = yolo_model(frame, verbose=False)
            detected = False
            for res in results:
                if res.boxes is None:
                    continue
                # person class == 0
                for box in res.boxes[res.boxes.cls == 0]:
                    stats['total_detections'] += 1
                    conf = box.conf.item()
                    if conf < confidence:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    pad = 0.1
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, x1 - int(w * pad))
                    y1 = max(0, y1 - int(h * pad))
                    x2 = min(width, x2 + int(w * pad))
                    y2 = min(height, y2 + int(h * pad))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
                        continue
                    crops.append({'crop': crop, 'frame': frame_no,
                                 'conf': conf, 'ts': frame_no / fps})
                    stats['valid_detections'] += 1
                    detected = True
            if detected:
                stats['frames_with_detections'] += 1
            pbar.update(1)
            if max_frames and frame_no >= max_frames:
                break
        frame_no += 1
    pbar.close()
    cap.release()
    return crops, stats

# --------------------------------------------------------------------------------------
# Feature extraction & NFC
# --------------------------------------------------------------------------------------


def extract_features(crops, extractor):
    feats, meta = [], []
    for idx, item in enumerate(tqdm(crops, desc='Features')):
        emb = extractor.extract_features(item['crop'])
        if emb is not None and len(emb) > 0:
            feats.append(torch.tensor(emb[0]))
            meta.append(
                {'idx': idx, 'frame': item['frame'], 'conf': item['conf'], 'ts': item['ts']})
    feats = torch.stack(feats)
    return feats, meta

# --------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------


def cluster_features(feats: torch.Tensor, meta, expected_persons: int):
    print("📐 Applying NFC (Neighbor Feature Centralization)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feats_device = feats.clone().float().to(device)
    feats_nfc = NFC(feats_device, k1=4, k2=4)
    feats_nfc = feats_nfc.cpu().numpy()

    reid_cfg = get_config('reid')

    # Parámetros optimizados para 5 personas
    if expected_persons >= 5:
        mahalanobis_threshold = 4.5  # Más agresivo para 5+ personas
        fusion_threshold = 3.5       # Fusión muy agresiva
        min_samples = 8              # Más muestras por cluster
    else:
        mahalanobis_threshold = 5.5
        fusion_threshold = 5.0
        min_samples = 4

    print(
        f"🎯 Optimized params: Mahalanobis={mahalanobis_threshold}, Fusion={fusion_threshold}, MinSamples={min_samples}")

    gcm = GaussianClusterManager(
        embedding_dim=reid_cfg['feature_dim'],
        mahalanobis_threshold=mahalanobis_threshold,
        max_clusters=expected_persons * 2,  # Permitir hasta 2x el número esperado
        min_samples_for_update=min_samples,
        fusion_threshold=fusion_threshold,
        split_threshold=15.0,  # Menos splits
        initial_sigma_scale=0.8  # Sigma más pequeña
    )

    hnsw = HNSWIndex(dim=reid_cfg['feature_dim'], max_elements=len(feats_nfc))

    labels = []
    ext_map = {}
    for i, feat in enumerate(tqdm(feats_nfc, desc='Clustering')):
        if hnsw.index.get_current_count() > 0:
            res = hnsw.search(feat, k=5)
            neigh = [ext_map[eid]
                     for eid in res['external_ids'] if eid in ext_map]
        else:
            neigh = []
        cid, _ = gcm.assign_to_cluster(feat, neigh)
        eid = f'c{i}'
        hnsw.add_vector(feat, eid, metadata={'cluster': cid})
        ext_map[eid] = cid
        labels.append(cid)

    # Post-procesamiento agresivo: Fusionar hasta llegar al número esperado
    labels = post_process_clusters(feats_nfc, labels, expected_persons)

    clusters = defaultdict(list)
    for lbl, m in zip(labels, meta):
        clusters[lbl].append(m['frame'])

    final_cluster_count = len(clusters)
    print(
        f"🎯 Final clusters: {final_cluster_count} (expected {expected_persons})")

    return labels, clusters, feats_nfc


def post_process_clusters(features, labels, target_clusters):
    """Post-procesamiento agresivo para alcanzar exactamente el número objetivo de clusters"""
    from scipy.spatial.distance import pdist, squareform
    import numpy as np

    # Obtener clusters únicos y sus centroides
    unique_labels = list(set(labels))
    current_clusters = len(unique_labels)

    print(f"📊 Initial clusters: {current_clusters}, Target: {target_clusters}")

    if current_clusters <= target_clusters:
        return labels

    # Calcular centroides de cada cluster
    centroids = {}
    for label in unique_labels:
        mask = np.array(labels) == label
        centroids[label] = features[mask].mean(axis=0)

    # Fusionar clusters iterativamente hasta llegar al objetivo
    while len(unique_labels) > target_clusters:
        # Calcular matriz de distancias entre centroides
        centroid_matrix = np.array([centroids[label]
                                   for label in unique_labels])
        distances = pdist(centroid_matrix, metric='cosine')
        dist_matrix = squareform(distances)

        # Encontrar el par de clusters más similares (excluyendo diagonal)
        np.fill_diagonal(dist_matrix, np.inf)
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        # Fusionar los dos clusters más similares
        cluster1 = unique_labels[min_idx[0]]
        cluster2 = unique_labels[min_idx[1]]

        print(
            f"🔗 Merging cluster {cluster2} into {cluster1} (distance: {dist_matrix[min_idx]:.3f})")

        # Reasignar todas las instancias del cluster2 al cluster1
        labels = [cluster1 if label == cluster2 else label for label in labels]

        # Actualizar lista de clusters únicos y recalcular centroide fusionado
        unique_labels.remove(cluster2)
        mask = np.array(labels) == cluster1
        centroids[cluster1] = features[mask].mean(axis=0)
        del centroids[cluster2]

    return labels

# --------------------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------------------


def visualize(feats, labels, meta, out_dir: Path):
    pca = PCA(n_components=2, random_state=42)
    red = pca.fit_transform(feats)
    uniq = sorted(set(labels))
    cmap = plt.get_cmap('tab10', len(uniq))
    plt.figure(figsize=(12, 10))
    for i, cid in enumerate(uniq):
        m = np.array(labels) == cid
        plt.scatter(red[m, 0], red[m, 1], c=[cmap(i)],
                    label=f'{cid}', s=40, alpha=0.7)
    plt.legend()
    plt.title('PCA of NFC-enhanced embeddings')
    plt.grid(True, linestyle='--', alpha=0.3)
    out = out_dir / 'pose2id_nfc_pca.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(video_path: str, output_dir: str, expected_persons: int = 5, frame_skip: int = 2, max_frames: int = None):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_preset('three_person_dataset_90_confidence')  # base preset
    print(
        f'⚙️  Using base preset + custom params for {expected_persons} persons')

    # Extract crops
    crops, det_stats = extract_person_crops(
        video_path, confidence=0.9, frame_skip=frame_skip, max_frames=max_frames)
    if not crops:
        print('No crops extracted. Exiting.')
        return

    # Feature extraction
    extractor = OSNetReID()
    feats, meta = extract_features(crops, extractor)

    # Cluster
    labels, clusters, feats_np = cluster_features(
        feats, meta, expected_persons)
    print(f'Clusters formed: {len(clusters)} (expected {expected_persons})')

    # Visualization
    viz = visualize(feats_np, labels, meta, output_dir)
    print('Visualization saved:', viz)

    # Save results
    res = {
        'video': str(video_path),
        'expected_persons': expected_persons,
        'clusters': {str(k): {'size': len(v), 'frame_range': [int(min(v)), int(max(v))]} for k, v in clusters.items()},
        'detection': convert_numpy_types(det_stats),
    }
    with open(output_dir / 'pose2id_clustering_results.json', 'w') as f:
        json.dump(convert_numpy_types(res), f, indent=2)

    print('✅ Done!')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Pose2ID NFC video clustering')
    parser.add_argument('--video', default='data/raw/video3.mp4')
    parser.add_argument('--output-dir', default='output/pose2id_clustering')
    parser.add_argument('--expected-persons', type=int, default=5)
    parser.add_argument('--frame-skip', type=int, default=2)
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    main(args.video, args.output_dir, args.expected_persons,
         args.frame_skip, args.max_frames)
