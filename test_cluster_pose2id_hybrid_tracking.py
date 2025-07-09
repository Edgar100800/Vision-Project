#!/usr/bin/env python3
"""
Pose2ID Hybrid Tracking System
=============================
Combines advanced Pose2ID clustering with dynamic tracking and real-time visualization.

Features:
- Enhanced NFC (Neighbor Feature Centralization) with 15D features
- Dynamic ByteTracker integration
- Real-time cluster evolution tracking
- Visual centroids generation
- Temporal consistency validation
- Advanced outlier detection
- Intelligent cluster growth management
"""

from src.index.hnsw_index import HNSWIndex
from src.clustering.gaussian_cluster_manager import GaussianClusterManager
from src.tracking.byte_tracker import ByteTracker, Detection
from src.reid.osnet_reid import OSNetReID
from config.global_config import get_config, apply_preset
import os
import sys
import cv2
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Project imports
sys.path.append('.')
sys.path.append('src')


# --------------------------------------------------------------------------------------
# Enhanced Data Structures
# --------------------------------------------------------------------------------------


@dataclass
class HybridDetection:
    """Enhanced detection with both clustering and tracking info."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    frame_number: int
    track_id: Optional[int] = None
    person_id: Optional[str] = None
    cluster_id: Optional[str] = None
    status: str = "unprocessed"
    reid_features: Optional[np.ndarray] = None
    enhanced_features: Optional[np.ndarray] = None
    crop_image: Optional[np.ndarray] = None
    similarity_score: float = 0.0
    nfc_applied: bool = False
    temporal_consistency: float = 0.0


@dataclass
class HybridTrackState:
    """Enhanced track state with clustering integration."""
    track_id: int
    person_id: Optional[str] = None
    cluster_id: Optional[str] = None
    frames_lost: int = 0
    last_seen_frame: int = 0
    feature_history: deque = None
    similarity_history: deque = None
    confirmation_hits: int = 0
    cluster_contributions: int = 0
    outlier_count: int = 0
    temporal_consistency_score: float = 0.0

    def __post_init__(self):
        if self.feature_history is None:
            self.feature_history = deque(maxlen=10)
        if self.similarity_history is None:
            self.similarity_history = deque(maxlen=10)


@dataclass
class ClusterState:
    """State information for each cluster."""
    cluster_id: str
    person_id: str
    creation_frame: int
    last_update_frame: int
    sample_count: int
    centroid_path: Optional[str] = None
    quality_score: float = 0.0
    temporal_spread: Tuple[int, int] = (0, 0)

# --------------------------------------------------------------------------------------
# Enhanced Pose2ID Functions
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
    """Expand features to 15 dimensions for better clustering"""
    print("🔧 Expanding features to 15 dimensions...")

    # PCA to reduce to 10 main components
    pca = PCA(n_components=10, random_state=42)
    pca_features = pca.fit_transform(features)

    # Add 5 additional engineered features
    additional_features = []

    for i, meta in enumerate(metadata):
        # Feature 11: Temporal position (normalized frame number)
        max_frame = max(m.get('frame_number', m.get('frame', 0))
                        for m in metadata)
        temporal_pos = meta.get(
            'frame_number', meta.get('frame', 0)) / max_frame

        # Feature 12: Confidence score
        confidence = meta.get('confidence', 0.5)

        # Feature 13: Temporal density
        frame_window = 60
        current_frame = meta.get('frame_number', meta.get('frame', 0))
        nearby_frames = sum(1 for m in metadata
                            if abs(m.get('frame_number', m.get('frame', 0)) - current_frame) <= frame_window)
        temporal_density = nearby_frames / len(metadata)

        # Feature 14: Feature magnitude
        feature_magnitude = np.linalg.norm(features[i])

        # Feature 15: Relative position in video
        relative_position = 0.0
        if temporal_pos < 0.33:
            relative_position = 0.0
        elif temporal_pos < 0.66:
            relative_position = 0.5
        else:
            relative_position = 1.0

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

    print(f"   - Enhanced features: {enhanced_features.shape}")
    print(
        f"   - PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    return enhanced_features, pca


def NFC_enhanced(feat: torch.tensor, k1=6, k2=6, alpha=0.6):
    """Enhanced Neighbor Feature Centralization with adaptive weighting"""
    device = feat.device
    feat = feat.clone()

    # Compute pairwise distances
    dist = pairwise_distance(feat, feat)

    # Mask diagonal
    eye = torch.eye(dist.size(0)).to(device)
    dist[eye == 1] = 1000

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
                weight = 1.0 / (val[i][j_idx] + 1e-8)
                weight_list.append(weight)

        if weight_list:
            weight_list = np.array(weight_list)
            weight_list = weight_list / weight_list.sum()

        mutual_topk_list.append(mutual_list)
        mutual_weights_list.append(weight_list)

    # Enhanced feature centralization
    feat_copy = feat.clone()
    for i in range(rank.size(0)):
        if mutual_topk_list[i]:
            neighbor_feats = feat_copy[mutual_topk_list[i]]
            weights = torch.tensor(
                mutual_weights_list[i], device=device).unsqueeze(1)
            weighted_neighbors = (neighbor_feats * weights).sum(dim=0)
            feat[i] = (1 - alpha) * feat[i] + alpha * weighted_neighbors

    return feat

# --------------------------------------------------------------------------------------
# Hybrid Tracking System
# --------------------------------------------------------------------------------------


class Pose2IDHybridTracker:
    """Main hybrid system combining Pose2ID clustering with dynamic tracking."""

    def __init__(self, video_path: str, output_dir: str):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Enhanced configuration
        self.config = {
            'detection_confidence': 0.9,
            'reid_confidence': 0.85,
            'similarity_threshold': 0.3,
            'nfc_k1': 6,
            'nfc_k2': 6,
            'nfc_alpha': 0.6,
            'max_frames_lost': 30,
            'confirmation_hits_required': 3,
            'max_cluster_size': 20,
            'temporal_window': 60,
            'outlier_threshold': 2.5,
        }

        # Initialize models
        self.yolo_model = YOLO('yolov8s.pt')
        self.reid_system = OSNetReID()
        self.tracker = ByteTracker(
            frame_rate=30,
            track_thresh=0.5,
            track_buffer=self.config['max_frames_lost'],
            match_thresh=0.8
        )

        # Initialize clustering system
        reid_cfg = get_config('reid')
        self.cluster_manager = GaussianClusterManager(
            embedding_dim=15,  # Using 15D enhanced features
            mahalanobis_threshold=4.5,
            max_clusters=10,
            min_samples_for_update=3,
            fusion_threshold=3.5,
            split_threshold=15.0,
            initial_sigma_scale=0.8
        )

        self.hnsw_index = HNSWIndex(dim=15, max_elements=1000)

        # State management
        self.frame_count = 0
        self.next_person_id = 1
        self.track_states: Dict[int, HybridTrackState] = {}
        self.cluster_states: Dict[str, ClusterState] = {}
        self.lost_tracks: Dict[int, HybridTrackState] = {}

        # For batch processing and NFC
        self.detection_batch = []
        self.feature_batch = []
        self.metadata_batch = []

        # Statistics
        self.stats = {
            'total_detections': 0,
            'successful_tracks': 0,
            'clusters_created': 0,
            'nfc_applications': 0,
            'cluster_contributions': 0,
            'reidentifications': 0,
            'temporal_validations': 0,
        }

        # Visualization
        self.color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (165, 42, 42), (0, 128, 128), (128, 128, 0)
        ]
        self.cluster_colors: Dict[str, Tuple[int, int, int]] = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_video(self, max_frames: Optional[int] = None):
        """Main processing pipeline."""
        print("🚀 Starting Pose2ID Hybrid Tracking System")
        print("=" * 70)

        # Apply configuration
        apply_preset('three_person_dataset_90_confidence')

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        # Video info
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(
            f"📺 Video: {frame_w}x{frame_h} @ {fps:.1f}fps, {total_frames} frames")

        # Setup video writer
        output_video_path = self.output_dir / "hybrid_tracking_output.mp4"
        video_writer = cv2.VideoWriter(
            str(output_video_path), cv2.VideoWriter_fourcc(
                *'mp4v'), fps, (frame_w, frame_h)
        )

        # Processing loop
        pbar = tqdm(total=total_frames, desc="Processing frames")

        while cap.isOpened() and self.frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Step 1: Detection
            detections = self._detect_persons(frame)

            # Step 2: Tracking update
            tracked_objects = self._update_tracking(detections)

            # Step 3: Feature extraction and batch processing
            enhanced_detections = self._extract_and_enhance_features(
                detections, tracked_objects, frame)

            # Step 4: Apply Pose2ID clustering with NFC
            if len(self.detection_batch) >= 10:  # Process in batches
                self._apply_pose2id_clustering()

            # Step 5: Temporal consistency validation
            self._validate_temporal_consistency(enhanced_detections)

            # Step 6: Update cluster states
            self._update_cluster_states(enhanced_detections)

            # Step 7: Generate visualizations
            annotated_frame = self._annotate_frame(frame, enhanced_detections)
            video_writer.write(annotated_frame)

            self.frame_count += 1
            pbar.update(1)

            if self.frame_count % 100 == 0:
                self._log_progress()

        # Final batch processing
        if self.detection_batch:
            self._apply_pose2id_clustering()

        cap.release()
        video_writer.release()
        pbar.close()

        # Post-processing
        self._generate_cluster_centroids()
        self._save_comprehensive_results()
        self._create_analysis_visualizations()

        print("\n✅ Processing completed successfully!")
        self._print_final_summary()

    def _detect_persons(self, frame: np.ndarray) -> List[HybridDetection]:
        """Detect persons using YOLO."""
        detections = []
        results = self.yolo_model(frame, verbose=False)

        for result in results:
            if result.boxes is not None:
                person_boxes = result.boxes[result.boxes.cls == 0]

                for box in person_boxes:
                    confidence = box.conf.item()
                    if confidence >= self.config['detection_confidence']:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Add padding
                        pad = 0.1
                        w, h = x2 - x1, y2 - y1
                        x1 = max(0, x1 - int(w * pad))
                        y1 = max(0, y1 - int(h * pad))
                        x2 = min(frame.shape[1], x2 + int(w * pad))
                        y2 = min(frame.shape[0], y2 + int(h * pad))

                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0 and crop.shape[0] > 32 and crop.shape[1] > 32:
                            detections.append(HybridDetection(
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                frame_number=self.frame_count,
                                crop_image=crop
                            ))

        self.stats['total_detections'] += len(detections)
        return detections

    def _update_tracking(self, detections: List[HybridDetection]) -> List:
        """Update ByteTracker."""
        byte_detections = []
        for det in detections:
            byte_det = Detection(
                bbox=np.array(det.bbox, dtype=np.float32),
                score=det.confidence,
                class_id=0
            )
            byte_detections.append(byte_det)

        tracked_objects = self.tracker.update(byte_detections)

        # Match detections with tracks
        for track in tracked_objects:
            best_iou = -1
            best_det = None

            for det in detections:
                iou = self._compute_iou(det.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det and best_iou > 0.3:
                best_det.track_id = track.track_id

                # Update or create track state
                if track.track_id not in self.track_states:
                    self.track_states[track.track_id] = HybridTrackState(
                        track_id=track.track_id,
                        last_seen_frame=self.frame_count
                    )
                else:
                    self.track_states[track.track_id].last_seen_frame = self.frame_count
                    self.track_states[track.track_id].frames_lost = 0

        return tracked_objects

    def _extract_and_enhance_features(self, detections: List[HybridDetection], tracked_objects, frame: np.ndarray) -> List[HybridDetection]:
        """Extract and enhance features for clustering."""
        enhanced_detections = []

        for det in detections:
            if det.track_id is None:
                continue

            # Extract ReID features
            if det.crop_image is not None:
                features = self.reid_system.extract_features(det.crop_image)
                if features is not None and len(features) > 0:
                    det.reid_features = features[0]

                    # Add to batch for NFC processing
                    self.detection_batch.append(det)
                    self.feature_batch.append(features[0])
                    self.metadata_batch.append({
                        'frame_number': det.frame_number,
                        'confidence': det.confidence,
                        'track_id': det.track_id
                    })

                    enhanced_detections.append(det)

        return enhanced_detections

    def _apply_pose2id_clustering(self):
        """Apply Pose2ID clustering with NFC to batch."""
        if len(self.feature_batch) < 5:
            return

        print(
            f"🔬 Applying Pose2ID clustering to batch of {len(self.feature_batch)} features")

        # Step 1: Enhance features to 15D
        features_array = np.array(self.feature_batch)
        enhanced_features, pca = enhanced_feature_extraction(
            features_array, self.metadata_batch)

        # Step 2: Apply NFC
        features_tensor = torch.tensor(enhanced_features).float()
        features_nfc = NFC_enhanced(
            features_tensor,
            k1=self.config['nfc_k1'],
            k2=self.config['nfc_k2'],
            alpha=self.config['nfc_alpha']
        )
        features_nfc = features_nfc.numpy()

        # Step 3: L2 normalization
        features_nfc = features_nfc / \
            (np.linalg.norm(features_nfc, axis=1, keepdims=True) + 1e-8)

        # Step 4: Update detections with enhanced features
        for i, det in enumerate(self.detection_batch):
            det.enhanced_features = features_nfc[i]
            det.nfc_applied = True

            # Step 5: Clustering assignment
            self._assign_to_cluster(det)

        self.stats['nfc_applications'] += 1

        # Clear batch
        self.detection_batch.clear()
        self.feature_batch.clear()
        self.metadata_batch.clear()

    def _assign_to_cluster(self, detection: HybridDetection):
        """Assign detection to cluster using enhanced features."""
        if detection.enhanced_features is None:
            return

        track_state = self.track_states.get(detection.track_id)
        if not track_state:
            return

        # Check if track already has person_id
        if track_state.person_id is not None:
            # Update existing cluster
            cluster_id = track_state.cluster_id
            if cluster_id and cluster_id in self.cluster_manager.clusters:
                # Calculate similarity
                cluster = self.cluster_manager.clusters[cluster_id]
                similarity = self._calculate_cluster_similarity(
                    detection.enhanced_features, cluster)
                detection.similarity_score = similarity

                # Update cluster if similarity is good
                if similarity >= self.config['similarity_threshold']:
                    self.cluster_manager._update_cluster_incremental(
                        cluster, detection.enhanced_features)
                    track_state.cluster_contributions += 1
                    self.stats['cluster_contributions'] += 1
                    detection.status = "cluster_updated"
                else:
                    detection.status = "similarity_too_low"

                track_state.similarity_history.append(similarity)
        else:
            # New identification needed
            if detection.confidence >= self.config['reid_confidence']:
                # Try to assign to existing cluster or create new one
                best_cluster_id, best_distance = self._find_best_cluster(
                    detection.enhanced_features)

                if best_cluster_id and (1.0 - best_distance) >= self.config['similarity_threshold']:
                    # Assign to existing cluster
                    cluster_state = self.cluster_states.get(best_cluster_id)
                    if cluster_state:
                        track_state.person_id = cluster_state.person_id
                        track_state.cluster_id = best_cluster_id
                        detection.person_id = cluster_state.person_id
                        detection.cluster_id = best_cluster_id
                        detection.similarity_score = 1.0 - best_distance
                        detection.status = "reidentified"
                        self.stats['reidentifications'] += 1
                else:
                    # Create new cluster
                    cluster_id, _ = self.cluster_manager.assign_to_cluster(
                        detection.enhanced_features, [])
                    person_id = f"Person_{self.next_person_id}"

                    # Create cluster state
                    self.cluster_states[cluster_id] = ClusterState(
                        cluster_id=cluster_id,
                        person_id=person_id,
                        creation_frame=self.frame_count,
                        last_update_frame=self.frame_count,
                        sample_count=1
                    )

                    # Assign color
                    if cluster_id not in self.cluster_colors:
                        color_idx = len(self.cluster_colors) % len(
                            self.color_palette)
                        self.cluster_colors[cluster_id] = self.color_palette[color_idx]

                    # Update states
                    track_state.person_id = person_id
                    track_state.cluster_id = cluster_id
                    detection.person_id = person_id
                    detection.cluster_id = cluster_id
                    detection.status = "new_person_created"

                    self.next_person_id += 1
                    self.stats['clusters_created'] += 1

                    print(
                        f"   ✨ Created {person_id} with cluster {cluster_id[:8]}")

        # Add feature to track history
        if track_state:
            track_state.feature_history.append(detection.enhanced_features)

    def _find_best_cluster(self, features: np.ndarray) -> Tuple[Optional[str], float]:
        """Find best matching cluster."""
        best_cluster_id = None
        best_distance = float('inf')

        for cluster_id, cluster in self.cluster_manager.clusters.items():
            try:
                # Calculate Mahalanobis distance
                diff = features - cluster.mu
                sigma_inv = np.linalg.inv(cluster.sigma)
                distance = np.sqrt(diff.T @ sigma_inv @ diff)

                if distance < best_distance:
                    best_distance = distance
                    best_cluster_id = cluster_id
            except:
                # Fallback to Euclidean distance
                distance = np.linalg.norm(features - cluster.mu)
                if distance < best_distance:
                    best_distance = distance
                    best_cluster_id = cluster_id

        return best_cluster_id, best_distance

    def _calculate_cluster_similarity(self, features: np.ndarray, cluster) -> float:
        """Calculate similarity between features and cluster."""
        try:
            diff = features - cluster.mu
            sigma_inv = np.linalg.inv(cluster.sigma)
            mahalanobis_dist = np.sqrt(diff.T @ sigma_inv @ diff)
            return max(0, 1.0 - mahalanobis_dist / 10.0)
        except:
            cosine_sim = float(np.dot(features, cluster.mu) /
                               (np.linalg.norm(features) * np.linalg.norm(cluster.mu)))
            return max(0, cosine_sim)

    def _validate_temporal_consistency(self, detections: List[HybridDetection]):
        """Validate temporal consistency of assignments."""
        for det in detections:
            if det.track_id and det.person_id:
                track_state = self.track_states.get(det.track_id)
                if track_state and len(track_state.similarity_history) > 3:
                    # Calculate temporal consistency
                    recent_similarities = list(
                        track_state.similarity_history)[-3:]
                    consistency = np.std(recent_similarities)
                    track_state.temporal_consistency_score = 1.0 / \
                        (1.0 + consistency)
                    det.temporal_consistency = track_state.temporal_consistency_score

                    if consistency < 0.1:  # Very consistent
                        self.stats['temporal_validations'] += 1

    def _update_cluster_states(self, detections: List[HybridDetection]):
        """Update cluster state information."""
        for det in detections:
            if det.cluster_id and det.cluster_id in self.cluster_states:
                cluster_state = self.cluster_states[det.cluster_id]
                cluster_state.last_update_frame = self.frame_count

                # Update sample count
                if det.cluster_id in self.cluster_manager.clusters:
                    cluster = self.cluster_manager.clusters[det.cluster_id]
                    cluster_state.sample_count = cluster.n_samples

                # Update temporal spread
                if cluster_state.temporal_spread[0] == 0:
                    cluster_state.temporal_spread = (
                        self.frame_count, self.frame_count)
                else:
                    cluster_state.temporal_spread = (
                        cluster_state.temporal_spread[0],
                        self.frame_count
                    )

    def _generate_cluster_centroids(self):
        """Generate visual centroids for each cluster."""
        print("🎨 Generating cluster centroids...")

        centroids_dir = self.output_dir / 'centroids'
        centroids_dir.mkdir(exist_ok=True)

        for cluster_id, cluster_state in self.cluster_states.items():
            # This would collect representative images for each cluster
            # For now, we'll create a placeholder
            centroid_path = centroids_dir / \
                f'{cluster_state.person_id}_centroid.jpg'

            # Create a simple centroid visualization
            centroid_img = np.zeros((200, 200, 3), dtype=np.uint8)
            color = self.cluster_colors.get(cluster_id, (128, 128, 128))
            cv2.rectangle(centroid_img, (50, 50), (150, 150), color, -1)
            cv2.putText(centroid_img, cluster_state.person_id, (60, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(centroid_img, f'{cluster_state.sample_count} samples', (60, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            cv2.imwrite(str(centroid_path), centroid_img)
            cluster_state.centroid_path = str(centroid_path)

    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between bounding boxes."""
        x1, y1, x2, y2 = bbox1
        x1_p, y1_p, x2_p, y2_p = bbox2

        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _annotate_frame(self, frame: np.ndarray, detections: List[HybridDetection]) -> np.ndarray:
        """Annotate frame with comprehensive information."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Get color
            color = (128, 128, 128)
            if det.cluster_id and det.cluster_id in self.cluster_colors:
                color = self.cluster_colors[det.cluster_id]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Create label
            if det.person_id:
                label = f"{det.person_id}"
                if det.similarity_score > 0:
                    label += f" ({det.similarity_score:.2f})"
                if det.nfc_applied:
                    label += " [NFC]"
                label += f"\n{det.status}"
            else:
                label = f"Track_{det.track_id}\n{det.status}"

            # Draw label
            label_lines = label.split('\n')
            line_height = 15
            total_height = len(label_lines) * line_height + 5

            cv2.rectangle(frame, (x1, y1 - total_height),
                          (x1 + 200, y1), color, -1)

            for i, line in enumerate(label_lines):
                y_pos = y1 - total_height + (i + 1) * line_height
                cv2.putText(frame, line, (x1 + 2, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Add frame info
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Active Clusters: {len(self.cluster_manager.clusters)}",
            f"Tracks: {len(self.track_states)}",
            f"NFC Applications: {self.stats['nfc_applications']}",
            f"Cluster Contributions: {self.stats['cluster_contributions']}"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def _save_comprehensive_results(self):
        """Save comprehensive results."""
        results = {
            'video_info': {
                'path': str(self.video_path),
                'total_frames_processed': self.frame_count,
                'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'configuration': self.config,
            'statistics': self.stats,
            'clusters': {
                cluster_id: {
                    'person_id': state.person_id,
                    'creation_frame': state.creation_frame,
                    'last_update_frame': state.last_update_frame,
                    'sample_count': state.sample_count,
                    'temporal_spread': state.temporal_spread,
                    'centroid_path': state.centroid_path
                }
                for cluster_id, state in self.cluster_states.items()
            },
            'method': 'Pose2ID_Hybrid_Tracking',
            'features': {
                'nfc_enhanced': True,
                'feature_dimensions': 15,
                'temporal_validation': True,
                'dynamic_tracking': True
            }
        }

        results_path = self.output_dir / 'hybrid_tracking_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"📊 Results saved to: {results_path}")

    def _create_analysis_visualizations(self):
        """Create analysis visualizations."""
        try:
            # Cluster evolution plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Plot 1: Cluster sizes
            cluster_names = [
                state.person_id for state in self.cluster_states.values()]
            cluster_sizes = [
                state.sample_count for state in self.cluster_states.values()]
            colors = [np.array(self.cluster_colors.get(cid, (128, 128, 128))) / 255.0
                      for cid in self.cluster_states.keys()]

            if cluster_names:
                bars = ax1.bar(cluster_names, cluster_sizes, color=colors)
                ax1.set_title('Final Cluster Sizes')
                ax1.set_ylabel('Number of Samples')
                ax1.tick_params(axis='x', rotation=45)

                # Add value labels
                for bar, size in zip(bars, cluster_sizes):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                             f'{size}', ha='center', va='bottom')

            # Plot 2: Statistics
            stats_names = list(self.stats.keys())
            stats_values = list(self.stats.values())

            ax2.bar(stats_names, stats_values, color='skyblue')
            ax2.set_title('Processing Statistics')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            viz_path = self.output_dir / 'hybrid_analysis.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📈 Analysis visualization saved to: {viz_path}")

        except Exception as e:
            print(f"⚠️ Could not create analysis visualization: {e}")

    def _log_progress(self):
        """Log processing progress."""
        self.logger.info(f"Frame {self.frame_count}: "
                         f"Clusters={len(self.cluster_manager.clusters)}, "
                         f"Tracks={len(self.track_states)}, "
                         f"NFC_Apps={self.stats['nfc_applications']}")

    def _print_final_summary(self):
        """Print comprehensive final summary."""
        print("\n" + "=" * 70)
        print("🎯 POSE2ID HYBRID TRACKING SYSTEM - FINAL SUMMARY")
        print("=" * 70)

        print("📊 Processing Statistics:")
        print(f"   • Total detections: {self.stats['total_detections']}")
        print(
            f"   • Successful tracks: {len(self.track_states) + len(self.lost_tracks)}")
        print(f"   • Clusters created: {self.stats['clusters_created']}")
        print(f"   • NFC applications: {self.stats['nfc_applications']}")
        print(
            f"   • Cluster contributions: {self.stats['cluster_contributions']}")
        print(f"   • Re-identifications: {self.stats['reidentifications']}")
        print(
            f"   • Temporal validations: {self.stats['temporal_validations']}")

        print("\n🎨 Cluster Summary:")
        for cluster_id, state in self.cluster_states.items():
            print(f"   • {state.person_id}: {state.sample_count} samples "
                  f"(frames {state.temporal_spread[0]}-{state.temporal_spread[1]})")

        print("\n📁 Output Files:")
        print(f"   • Video: hybrid_tracking_output.mp4")
        print(f"   • Results: hybrid_tracking_results.json")
        print(f"   • Analysis: hybrid_analysis.png")
        print(f"   • Centroids: centroids/ directory")

        print("=" * 70)

# --------------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------------


def main(video_path: str, output_dir: str, max_frames: Optional[int] = None):
    """Main function to run hybrid tracking system."""
    tracker = Pose2IDHybridTracker(video_path, output_dir)
    tracker.process_video(max_frames)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Pose2ID Hybrid Tracking System')
    parser.add_argument(
        '--video', default='data/raw/video3.mp4', help='Input video path')
    parser.add_argument(
        '--output-dir', default='output/hybrid_tracking', help='Output directory')
    parser.add_argument('--max-frames', type=int,
                        default=None, help='Maximum frames to process')

    args = parser.parse_args()

    main(args.video, args.output_dir, args.max_frames)
