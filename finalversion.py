#!/usr/bin/env python3
"""
Final Version: Advanced Person Re-ID with Pose2ID Integration
Integrates Pose2ID's NFC (Neighbor Feature Centralization) with our tracking system
"""

from tracking.byte_tracker import ByteTracker, Detection
from reid.osnet_reid import OSNetReID
import os
import sys
import cv2
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import List, Optional, Tuple, Dict, Any
import time
from tqdm import tqdm
import json

# Import YOLO and ByteTracker
from ultralytics import YOLO

# Import our existing modules
sys.path.append('src')

# Pose2ID Functions (embedded)


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
    feat = feat.clone()

    # If we have fewer features than k1, adjust k1 and k2
    n_features = feat.size(0)
    if n_features <= 1:
        return feat  # Can't apply NFC with only 1 feature

    k1 = min(k1, n_features - 1)  # Ensure k1 is less than number of features
    k2 = min(k2, n_features - 1)  # Ensure k2 is less than number of features

    dist = pairwise_distance(feat.to('cuda' if torch.cuda.is_available() else 'cpu'),
                             feat.to('cuda' if torch.cuda.is_available() else 'cpu')).to('cpu')

    eye = torch.eye(dist.size(0)).to(dist.device)
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


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def ID2(feats, pid):
    from torch.nn import functional as F

    # Ensure feats is 2D (N, feature_dim)
    if len(feats.shape) == 3:
        feats = feats.squeeze(1)  # Remove middle dimension if present

    feats = F.normalize(feats, dim=1, p=2)
    pids = np.asarray(pid)

    id_set = set(pids)
    id_list = list(id_set)
    id_list.sort()
    id_center = []
    for i in id_list:
        mask = torch.tensor(pids == i)
        if torch.sum(mask) > 0:  # Check if mask has any True values
            x = feats[mask].mean(dim=0)
            id_center.append(x)

    density = torch.zeros(feats.size(0))
    idx = 0
    for i in id_list:
        mask = torch.tensor(pids == i)
        if torch.sum(mask) > 0 and idx < len(id_center):
            center = id_center[idx].unsqueeze(0)
            try:
                dist = euclidean_distance(feats[mask], center)
                if dist.ndim > 1:
                    dist = dist.squeeze()
                density[mask] = torch.tensor(dist)
            except Exception as e:
                # If there's an error, set density to 0
                density[mask] = 0.0
            idx += 1
    return density


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """Enhanced person detection with Pose2ID features."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    frame_number: int
    track_id: Optional[int] = None
    person_id: Optional[str] = None
    status: str = "unprocessed"
    reid_features: Optional[np.ndarray] = None
    nfc_features: Optional[np.ndarray] = None  # NFC enhanced features
    crop_image: Optional[np.ndarray] = None
    similarity_score: float = 0.0
    id2_density: float = 0.0  # ID2 density metric


@dataclass
class TrackState:
    """Enhanced track state with Pose2ID metrics."""
    track_id: int
    person_id: Optional[str] = None
    frames_lost: int = 0
    last_seen_frame: int = 0
    feature_history: deque = None
    nfc_contributions: int = 0
    id2_scores: deque = None

    def __post_init__(self):
        if self.feature_history is None:
            self.feature_history = deque(maxlen=20)
        if self.id2_scores is None:
            self.id2_scores = deque(maxlen=10)


@dataclass
class Pose2IDCluster:
    """Cluster representation using Pose2ID methodology."""
    cluster_id: str
    person_id: str
    features: List[np.ndarray]
    centralized_features: Optional[np.ndarray] = None
    id2_density: float = 0.0
    size: int = 0

    def update_centralized_features(self, k1=2, k2=2):
        """Update centralized features using NFC."""
        if len(self.features) > 0:
            # Ensure all features have the same shape
            processed_features = []
            for f in self.features:
                if isinstance(f, np.ndarray):
                    if f.ndim == 1:
                        processed_features.append(
                            torch.tensor(f, dtype=torch.float32))
                    else:
                        processed_features.append(torch.tensor(
                            f.flatten(), dtype=torch.float32))
                else:
                    processed_features.append(torch.tensor(
                        f, dtype=torch.float32).flatten())

            feat_tensor = torch.stack(processed_features)
            centralized = NFC(feat_tensor, k1=k1, k2=k2).mean(dim=0)
            self.centralized_features = centralized.numpy()
            self.size = len(self.features)


class Pose2IDProcessor:
    """Advanced Person Re-ID Processor using Pose2ID methodology."""

    def __init__(self, input_video: str, output_dir: str):
        self.input_video = input_video
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Enhanced configuration for Pose2ID
        self.config = {
            'bbox_conf_threshold': 0.5,
            'person_conf_threshold': 0.85,
            'nfc_k1': 2,  # NFC parameter
            'nfc_k2': 2,  # NFC parameter
            'similarity_threshold': 0.6,
            'max_frames_lost': 15,
            'feature_buffer_size': 50,
            'id2_update_interval': 10,  # Update ID2 every N frames
            'min_features_for_nfc': 3,  # Minimum features needed for NFC
        }

        # Initialize systems
        self._initialize_systems()

        # Pose2ID specific data structures
        self.pose2id_clusters: Dict[str, Pose2IDCluster] = {}
        self.global_feature_buffer: List[np.ndarray] = []
        self.global_person_ids: List[str] = []
        self.feature_update_counter = 0

        # Statistics
        self.stats = {
            'total_detections': 0,
            'persons_created': 0,
            'nfc_centralizations': 0,
            'id2_computations': 0,
            'tracks_lost': 0,
            'tracks_reactivated': 0,
        }

        # Visualization
        self.track_colors = {}
        self.next_person_id = 1

    def _initialize_systems(self):
        """Initialize YOLO, ReID, and tracking systems."""
        # Initialize YOLO
        self.yolo_model = YOLO('yolov8n.pt')

        # Initialize ReID system
        self.reid_system = OSNetReID()

        # Initialize tracker
        self.tracker = ByteTracker()

        # Track states
        self.track_states: Dict[int, TrackState] = {}

        logger.info("Systems initialized successfully")

    def process_video(self, max_frames: Optional[int] = None):
        """Process video with Pose2ID integration."""
        cap = cv2.VideoCapture(self.input_video)

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(self.output_dir / 'pose2id_output.mp4'),
            fourcc, fps, (width, height)
        )

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_count >= max_frames:
                    break

                # Process frame
                processed_frame = self._process_frame(frame, frame_count)

                # Write frame
                out.write(processed_frame)

                frame_count += 1

                if frame_count % 50 == 0:
                    logger.info(f"Processed {frame_count} frames")

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        logger.info(f"Processing complete. Total frames: {frame_count}")
        self.print_summary()

    def _process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Process single frame with Pose2ID methodology."""
        # YOLO detection
        detections = self._detect_persons(frame, frame_number)

        # Update tracker
        tracked_objects = self._update_tracker(detections)

        # Update track states
        self._update_track_states(tracked_objects, frame_number)

        # Apply Pose2ID processing
        processed_detections = self._apply_pose2id_processing(
            detections, frame)

        # Update global feature buffer for NFC
        self._update_global_features(processed_detections)

        # Periodic ID2 computation
        if frame_number % self.config['id2_update_interval'] == 0:
            self._compute_id2_metrics()

        # Annotate frame
        annotated_frame = self._annotate_frame(frame, processed_detections)

        return annotated_frame

    def _detect_persons(self, frame: np.ndarray, frame_number: int) -> List[PersonDetection]:
        """Detect persons using YOLO."""
        results = self.yolo_model(
            frame, classes=[0], conf=self.config['bbox_conf_threshold'])
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()

                    # Extract crop
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]

                    detection = PersonDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(conf),
                        frame_number=frame_number,
                        crop_image=crop
                    )
                    detections.append(detection)

        self.stats['total_detections'] += len(detections)
        return detections

    def _update_tracker(self, detections: List[PersonDetection]) -> List:
        """Update ByteTracker with detections."""
        if not detections:
            return []

        # Convert to tracker format
        tracker_detections = []
        for d in detections:
            bbox = np.array([d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]])
            tracker_det = Detection(bbox=bbox, score=d.confidence, class_id=0)
            tracker_detections.append(tracker_det)

        # Update tracker
        tracked_objects = self.tracker.update(tracker_detections)

        # Match detections with tracks
        for detection in detections:
            best_match = None
            best_iou = 0.0

            for track in tracked_objects:
                iou = self._compute_iou(detection.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = track

            if best_match and best_iou > 0.3:
                detection.track_id = best_match.track_id

        return tracked_objects

    def _update_track_states(self, tracked_objects, frame_number: int):
        """Update track states and handle lost tracks."""
        active_track_ids = set()

        for track in tracked_objects:
            track_id = track.track_id
            active_track_ids.add(track_id)

            if track_id not in self.track_states:
                self.track_states[track_id] = TrackState(
                    track_id=track_id,
                    last_seen_frame=frame_number
                )
            else:
                self.track_states[track_id].last_seen_frame = frame_number
                self.track_states[track_id].frames_lost = 0

        # Handle lost tracks
        lost_tracks = []
        for track_id, track_state in self.track_states.items():
            if track_id not in active_track_ids:
                track_state.frames_lost += 1
                if track_state.frames_lost >= self.config['max_frames_lost']:
                    lost_tracks.append(track_id)

        # Remove lost tracks
        for track_id in lost_tracks:
            del self.track_states[track_id]
            self.stats['tracks_lost'] += 1

    def _apply_pose2id_processing(self, detections: List[PersonDetection], frame: np.ndarray) -> List[PersonDetection]:
        """Apply Pose2ID processing to detections."""
        processed_detections = []

        for detection in detections:
            if detection.track_id is None:
                continue

            track_state = self.track_states.get(detection.track_id)
            if not track_state:
                continue

            # Extract ReID features
            if detection.crop_image is not None and detection.crop_image.size > 0:
                features = self.reid_system.extract_features(
                    detection.crop_image)
                detection.reid_features = features / np.linalg.norm(features)

                # Add to track history
                track_state.feature_history.append(detection.reid_features)

                # Person identification using Pose2ID methodology
                if detection.confidence >= self.config['person_conf_threshold']:
                    self._handle_person_identification(detection, track_state)

                processed_detections.append(detection)

        return processed_detections

    def _handle_person_identification(self, detection: PersonDetection, track_state: TrackState):
        """Handle person identification using Pose2ID clustering."""
        if track_state.person_id is None:
            # New person identification
            person_id = self._assign_person_identity(detection, track_state)
            if person_id:
                track_state.person_id = person_id
                detection.person_id = person_id
                detection.status = "new_person_created"
                self.stats['persons_created'] += 1
                logger.info(
                    f"Created new person {person_id} for track {track_state.track_id}")
        else:
            # Existing person - update cluster
            detection.person_id = track_state.person_id
            detection.status = "person_tracked"
            self._update_pose2id_cluster(detection, track_state)

    def _assign_person_identity(self, detection: PersonDetection, track_state: TrackState) -> Optional[str]:
        """Assign person identity using Pose2ID similarity."""
        if detection.reid_features is None:
            return None

        # Check similarity with existing clusters
        best_cluster = None
        best_similarity = 0.0

        for cluster in self.pose2id_clusters.values():
            if cluster.centralized_features is not None:
                similarity = self._compute_similarity(
                    detection.reid_features,
                    cluster.centralized_features
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster

        # Assign to existing cluster if similarity is high enough
        if best_cluster and best_similarity > self.config['similarity_threshold']:
            return best_cluster.person_id

        # Create new cluster
        person_id = f"Person_{self.next_person_id}"
        self.next_person_id += 1

        cluster = Pose2IDCluster(
            cluster_id=f"cluster_{person_id}",
            person_id=person_id,
            features=[detection.reid_features.copy()]
        )
        cluster.update_centralized_features(
            k1=self.config['nfc_k1'],
            k2=self.config['nfc_k2']
        )

        self.pose2id_clusters[person_id] = cluster
        self._assign_track_color(person_id)

        return person_id

    def _update_pose2id_cluster(self, detection: PersonDetection, track_state: TrackState):
        """Update Pose2ID cluster with new features."""
        person_id = detection.person_id
        if person_id not in self.pose2id_clusters:
            return

        cluster = self.pose2id_clusters[person_id]

        # Add new feature (ensure it's 1D)
        feature = detection.reid_features.copy()
        if feature.ndim > 1:
            feature = feature.flatten()
        cluster.features.append(feature)

        # Keep only recent features
        if len(cluster.features) > self.config['feature_buffer_size']:
            cluster.features = cluster.features[-self.config['feature_buffer_size']:]

        # Update centralized features using NFC
        if len(cluster.features) >= self.config['min_features_for_nfc']:
            cluster.update_centralized_features(
                k1=self.config['nfc_k1'],
                k2=self.config['nfc_k2']
            )
            track_state.nfc_contributions += 1
            self.stats['nfc_centralizations'] += 1

    def _update_global_features(self, detections: List[PersonDetection]):
        """Update global feature buffer for ID2 computation."""
        for detection in detections:
            if detection.reid_features is not None and detection.person_id:
                # Ensure feature is stored as 1D array
                feature = detection.reid_features.copy()
                if feature.ndim > 1:
                    feature = feature.flatten()
                self.global_feature_buffer.append(feature)
                self.global_person_ids.append(detection.person_id)

        # Keep buffer size manageable
        max_buffer_size = 1000
        if len(self.global_feature_buffer) > max_buffer_size:
            self.global_feature_buffer = self.global_feature_buffer[-max_buffer_size:]
            self.global_person_ids = self.global_person_ids[-max_buffer_size:]

    def _compute_id2_metrics(self):
        """Compute ID2 density metrics."""
        if len(self.global_feature_buffer) < 10:
            return

        try:
            # Convert to tensor and ensure proper dimensions
            features_list = []
            for f in self.global_feature_buffer:
                if isinstance(f, np.ndarray):
                    if f.ndim == 1:
                        features_list.append(
                            torch.tensor(f, dtype=torch.float32))
                    else:
                        features_list.append(torch.tensor(
                            f.flatten(), dtype=torch.float32))
                else:
                    features_list.append(torch.tensor(
                        f, dtype=torch.float32).flatten())

            features = torch.stack(features_list)

            # Compute ID2 density
            densities = ID2(features, self.global_person_ids)

            # Update cluster densities
            for i, person_id in enumerate(self.global_person_ids):
                if person_id in self.pose2id_clusters and i < len(densities):
                    self.pose2id_clusters[person_id].id2_density = float(
                        densities[i])

            self.stats['id2_computations'] += 1

        except Exception as e:
            logger.warning(f"ID2 computation failed: {e}")

    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between features."""
        # Ensure features are 1D vectors
        feat1 = feat1.flatten()
        feat2 = feat2.flatten()

        # Avoid division by zero
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(feat1, feat2) / (norm1 * norm2))

    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bounding boxes."""
        x1, y1, x2, y2 = bbox1
        x1_t, y1_t, x2_t, y2_t = bbox2

        # Intersection
        xi1 = max(x1, x1_t)
        yi1 = max(y1, y1_t)
        xi2 = min(x2, x2_t)
        yi2 = min(y2, y2_t)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Union
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_t - x1_t) * (y2_t - y1_t)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _assign_track_color(self, person_id: str):
        """Assign unique color to track."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 128), (255, 165, 0), (255, 192, 203)
        ]
        self.track_colors[person_id] = colors[len(
            self.track_colors) % len(colors)]

    def _annotate_frame(self, frame: np.ndarray, detections: List[PersonDetection]) -> np.ndarray:
        """Annotate frame with tracking and Pose2ID information."""
        annotated = frame.copy()

        for detection in detections:
            if detection.person_id:
                color = self.track_colors.get(
                    detection.person_id, (255, 255, 255))
                x1, y1, x2, y2 = detection.bbox

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Prepare text
                text_lines = [
                    f"ID: {detection.person_id}",
                    f"Track: {detection.track_id}",
                    f"Conf: {detection.confidence:.2f}"
                ]

                # Add Pose2ID metrics
                if detection.person_id in self.pose2id_clusters:
                    cluster = self.pose2id_clusters[detection.person_id]
                    text_lines.append(f"Size: {cluster.size}")
                    text_lines.append(f"ID2: {cluster.id2_density:.3f}")

                # Draw text
                for i, text in enumerate(text_lines):
                    y_pos = y1 - 10 - (len(text_lines) - i - 1) * 15
                    cv2.putText(annotated, text, (x1, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Add frame info
        info_text = f"Frame: {detections[0].frame_number if detections else 0} | "
        info_text += f"Persons: {len(self.pose2id_clusters)} | "
        info_text += f"NFC: {self.stats['nfc_centralizations']}"

        cv2.putText(annotated, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated

    def print_summary(self):
        """Print processing summary."""
        logger.info("=" * 70)
        logger.info("POSE2ID PROCESSING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Detections: {self.stats['total_detections']}")
        logger.info(f"Persons Created: {self.stats['persons_created']}")
        logger.info(
            f"NFC Centralizations: {self.stats['nfc_centralizations']}")
        logger.info(f"ID2 Computations: {self.stats['id2_computations']}")
        logger.info(f"Tracks Lost: {self.stats['tracks_lost']}")
        logger.info(f"Tracks Reactivated: {self.stats['tracks_reactivated']}")
        logger.info("")
        logger.info("Cluster Details:")
        for person_id, cluster in self.pose2id_clusters.items():
            logger.info(
                f"  {person_id}: {cluster.size} features, ID2: {cluster.id2_density:.3f}")
        logger.info("=" * 70)

        # Save summary to file
        summary_data = {
            'stats': self.stats,
            'clusters': {
                person_id: {
                    'size': cluster.size,
                    'id2_density': cluster.id2_density
                }
                for person_id, cluster in self.pose2id_clusters.items()
            }
        }

        with open(self.output_dir / 'pose2id_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Pose2ID Person Re-Identification')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--max-frames', type=int,
                        help='Maximum frames to process')

    args = parser.parse_args()

    # Create processor
    processor = Pose2IDProcessor(args.video, args.output)

    # Process video
    processor.process_video(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
