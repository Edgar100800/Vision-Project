#!/usr/bin/env python3
"""
Advanced Video Person Re-ID with Dynamic Clustering and Visualization
====================================================================

This script implements a sophisticated person re-identification pipeline with:
- Dynamic cluster growth instead of simple centroid updates
- Real-time cluster visualization video generation
- Outlier guard and confirmation hits system
- Intelligent cluster purging when exceeding max size
- Integration with Gaussian clustering and HNSW indexing
"""
from config.global_config import get_config, apply_preset
from src.index.hnsw_index import HNSWIndex
from src.reid.osnet_reid import OSNetReID
from src.tracking.byte_tracker import ByteTracker, Detection
from src.clustering.hnsw_gaussian_integration import HNSWGaussianIntegration
from src.clustering.gaussian_cluster_manager import GaussianClusterManager
import os
import sys
import cv2
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

import numpy as np
from ultralytics import YOLO
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation

# Add project root to path to allow direct imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Logging Setup ---
log_dir = Path("output/test_demo_new_logic/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "reid_advanced.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """A dataclass to hold all information about a single person detection."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    frame_number: int
    track_id: Optional[int] = None
    person_id: Optional[str] = None
    status: str = "unprocessed"
    reid_features: Optional[np.ndarray] = None
    crop_image: Optional[np.ndarray] = None
    iou_score: float = 0.0
    similarity_score: float = 0.0


@dataclass
class TrackState:
    """Enhanced track state with confirmation hits and outlier detection."""
    track_id: int
    person_id: Optional[str] = None
    frames_lost: int = 0
    last_seen_frame: int = 0
    similarity_history: deque = None
    confirmation_hits: int = 0
    required_hits: int = 3
    outlier_count: int = 0
    max_outliers: int = 2
    cluster_contributions: int = 0

    def __post_init__(self):
        if self.similarity_history is None:
            self.similarity_history = deque(maxlen=10)


@dataclass
class ClusterVisualizationData:
    """Data for cluster visualization."""
    frame_number: int
    cluster_id: str
    centroid: np.ndarray
    samples: List[np.ndarray]
    size: int
    confidence_ellipse: Optional[np.ndarray] = None


class AdvancedPersonReIDProcessor:
    """Advanced Re-ID processor with dynamic clustering and visualization."""

    def __init__(self, input_video: str, output_dir: str):
        self.input_video = input_video
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Enhanced configuration following the new flowchart
        self.config = {
            'bbox_conf_threshold': 0.25,  # Reducido para más detecciones
            'person_conf_threshold': 0.85,  # Reducido aún más
            'sim_threshold_new_cluster': 0.15,  # Reducido para más clusters
            'sim_threshold_assign_existing': 0.3,  # Reducido para más asignaciones
            'sim_threshold_update_cluster': 0.3,  # Reducido para más updates
            'max_frames_lost': 30,  # Aumentado para más persistencia
            'similarity_frames_required': 1,
            'iou_threshold': 0.05,  # Reducido para matching más flexible
            'confirmation_hits_required': 1,  # Reducido para updates inmediatos
            'outlier_sigma_threshold': 5.0,  # Aumentado para menos outliers
            'max_cluster_size': 15,  # Aumentado para más muestras
            'purge_percentage': 0.1,  # Reducido para menos purgas
        }

        # Initialize models and components
        self.yolo_model = YOLO('yolov8s.pt')
        self.reid_system = OSNetReID()
        self.tracker = ByteTracker(
            frame_rate=30,
            track_thresh=0.5,
            track_buffer=self.config['max_frames_lost'],
            match_thresh=0.8,
            high_thresh=0.6,
            low_thresh=0.1
        )

        # Initialize advanced clustering system
        gaussian_config = {
            'initial_sigma_scale': 0.5,
            'mahalanobis_threshold': self.config['sim_threshold_assign_existing'],
            'min_samples_for_update': 2,
            'max_clusters': 50,
            'fusion_threshold': 1.5,
            'split_threshold': 8.0
        }

        self.clustering_system = HNSWGaussianIntegration(
            embedding_dim=512,
            gaussian_config=gaussian_config,
            top_k_candidates=5
        )

        # Enhanced state management
        self.next_person_id = 1
        self.track_states: Dict[int, TrackState] = {}
        self.lost_tracks: Dict[int, TrackState] = {}
        self.frame_count = 0

        # Cluster visualization data
        self.cluster_viz_data: List[ClusterVisualizationData] = []
        self.cluster_colors: Dict[str, Tuple[int, int, int]] = {}
        self.color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (165, 42, 42), (0, 128, 128), (128, 128, 0),
            (255, 140, 0), (50, 205, 50), (30, 144, 255), (220, 20, 60)
        ]

        # Enhanced statistics
        self.stats = {
            'total_detections': 0,
            'high_confidence_reid': 0,
            'new_persons': 0,
            'reidentified_persons': 0,
            'confirmation_hits_achieved': 0,
            'outliers_detected': 0,
            'cluster_contributions': 0,
            'cluster_purges': 0,
            'tracks_lost': 0,
            'tracks_reactivated': 0,
            'reid_reactivations': 0,
        }

    def process_video(self, max_frames: Optional[int] = None):
        """Main processing loop implementing the advanced flowchart."""
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {self.input_video}")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Setup video writers
        main_video_path = self.output_dir / "advanced_reid_output.mp4"
        cluster_video_path = self.output_dir / "cluster_visualization.mp4"

        main_writer = cv2.VideoWriter(
            str(main_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
        cluster_writer = cv2.VideoWriter(
            str(cluster_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 600))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and self.frame_count >= max_frames):
                break

            # Step 1: YOLO Detection with confidence filtering
            all_detections = self._parse_yolo_results(
                self.yolo_model(frame, verbose=False), self.frame_count, frame)

            # Step 2: ByteTracker update
            tracked_objects = self._update_tracker(all_detections)

            # Step 3: Update track states and handle lost tracks
            self._update_track_states(tracked_objects)

            # Step 4: Apply advanced flowchart logic
            processed_detections = self._apply_advanced_flowchart(
                all_detections, tracked_objects, frame)

            # Step 5: Generate visualizations
            annotated_frame = self._annotate_frame(frame, processed_detections)
            cluster_viz_frame = self._generate_cluster_visualization()

            # Step 6: Write frames
            main_writer.write(annotated_frame)
            cluster_writer.write(cluster_viz_frame)

            # Step 7: Store cluster data for visualization
            self._store_cluster_visualization_data()

            logger.info(
                f"Frame {self.frame_count}: Detections={len(all_detections)}, "
                f"Tracked={len(tracked_objects)}, Active_Clusters={len(self.clustering_system.cluster_manager.clusters)}")
            self.frame_count += 1

        cap.release()
        main_writer.release()
        cluster_writer.release()

        logger.info(
            f"Processing complete. Videos saved to {main_video_path} and {cluster_video_path}")
        self.print_summary()
        self._save_cluster_evolution_plot()

    def _parse_yolo_results(self, yolo_results, frame_number, frame) -> List[PersonDetection]:
        """Parse YOLO results with confidence filtering."""
        detections = []
        for result in yolo_results:
            person_boxes = result.boxes[result.boxes.cls == 0]
            for box in person_boxes:
                confidence = box.conf.item()
                if confidence >= self.config['bbox_conf_threshold']:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    crop_image = frame[y1:y2,
                                       x1:x2] if x2 > x1 and y2 > y1 else None
                    detections.append(PersonDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        frame_number=frame_number,
                        crop_image=crop_image
                    ))
        self.stats['total_detections'] += len(detections)
        return detections

    def _update_tracker(self, detections: List[PersonDetection]) -> List:
        """Update ByteTracker with detections."""
        byte_detections = []
        for det in detections:
            byte_det = Detection(
                bbox=np.array(det.bbox, dtype=np.float32),
                score=det.confidence,
                class_id=0
            )
            byte_detections.append(byte_det)
        return self.tracker.update(byte_detections)

    def _update_track_states(self, tracked_objects):
        """Update track states and handle lost tracks."""
        active_track_ids = {track.track_id for track in tracked_objects}

        # Update existing tracks
        for track in tracked_objects:
            if track.track_id not in self.track_states:
                self.track_states[track.track_id] = TrackState(
                    track_id=track.track_id,
                    last_seen_frame=self.frame_count
                )
            else:
                self.track_states[track.track_id].last_seen_frame = self.frame_count
                self.track_states[track.track_id].frames_lost = 0

        # Handle lost tracks
        newly_lost = []
        for track_id, track_state in list(self.track_states.items()):
            if track_id not in active_track_ids:
                track_state.frames_lost = self.frame_count - track_state.last_seen_frame

                if track_state.frames_lost >= self.config['max_frames_lost']:
                    self.lost_tracks[track_id] = track_state
                    del self.track_states[track_id]
                    newly_lost.append(track_id)
                    self.stats['tracks_lost'] += 1

        if newly_lost:
            logger.info(
                f"Lost tracks (≥{self.config['max_frames_lost']} frames): {newly_lost}")

    def _apply_advanced_flowchart(self, detections: List[PersonDetection], tracked_objects, frame: np.ndarray) -> List[PersonDetection]:
        """Apply the advanced flowchart logic with clustering."""
        processed_detections = []

        for track in tracked_objects:
            # Find matching detection
            best_match = self._find_best_detection_match(detections, track)
            if not best_match:
                continue

            best_match.track_id = track.track_id
            track_state = self.track_states.get(track.track_id)
            if not track_state:
                continue

            # Flowchart: ¿Ya tiene ID asignado?
            if track_state.person_id is not None:
                # YES - Continue tracking with embedding loop
                best_match.person_id = track_state.person_id
                best_match.status = "tracking_continued"
                self._handle_embedding_loop(best_match, track_state, frame)
            else:
                # NO - New identification needed
                if best_match.confidence >= self.config['person_conf_threshold']:
                    self.stats['high_confidence_reid'] += 1
                    self._handle_new_identification(
                        best_match, track_state, frame)
                else:
                    best_match.status = "low_confidence_discarded"

            processed_detections.append(best_match)

        return processed_detections

    def _handle_embedding_loop(self, detection: PersonDetection, track_state: TrackState, frame: np.ndarray):
        """Handle the embedding loop with confirmation hits and outlier guard."""
        if detection.crop_image is None or detection.crop_image.size == 0:
            detection.status = "no_crop_available"
            return

        # Extract embedding
        features = self.reid_system.extract_features(detection.crop_image)
        norm_features = features / np.linalg.norm(features)
        detection.reid_features = norm_features

        # Get cluster for similarity check
        cluster_id = self._get_cluster_id_from_person(track_state.person_id)
        if not cluster_id:
            return
            
        cluster = self.clustering_system.get_cluster_by_id(cluster_id)
        if not cluster:
            return

        # Calculate similarity and IoU
        similarity = self._calculate_similarity(norm_features, cluster)
        detection.similarity_score = similarity

        # Store in history
        track_state.similarity_history.append(similarity)

        # SOLUCIÓN: Agregar SIEMPRE al cluster si es el mismo track (MUY agresivo)
        # Esto garantiza que los clusters crezcan continuamente
        if similarity >= 0.05:  # Umbral EXTREMADAMENTE bajo
            # Flowchart: Outlier guard σ (MUY permisivo)
            if not self._outlier_guard_check(norm_features, cluster):
                # Flowchart: Agregar al clúster
                self._add_to_cluster(norm_features, track_state.person_id)
                track_state.cluster_contributions += 1
                self.stats['cluster_contributions'] += 1
                detection.status = "added_to_cluster_aggressive"
                
                # Flowchart: Clúster > MaxSize?
                if self._check_cluster_size_and_purge(track_state.person_id):
                    detection.status = "cluster_purged"
                    self.stats['cluster_purges'] += 1
            else:
                track_state.outlier_count += 1
                self.stats['outliers_detected'] += 1
                detection.status = "outlier_detected"
        else:
            detection.status = "similarity_too_low"
            
        # Mantener lógica original para confirmation hits
        if similarity >= self.config['sim_threshold_update_cluster']:
            track_state.confirmation_hits += 1
            if track_state.confirmation_hits >= self.config['confirmation_hits_required']:
                self.stats['confirmation_hits_achieved'] += 1
        else:
            track_state.confirmation_hits = max(
                0, track_state.confirmation_hits - 1)

    def _handle_new_identification(self, detection: PersonDetection, track_state: TrackState, frame: np.ndarray):
        """Handle new identification using advanced clustering."""
        if detection.crop_image is None or detection.crop_image.size == 0:
            detection.status = "no_crop_available"
            return

        # Extract embedding
        features = self.reid_system.extract_features(detection.crop_image)
        norm_features = features / np.linalg.norm(features)
        detection.reid_features = norm_features

        # Check for reactivated tracks first
        if self._check_reactivated_track(detection, track_state, norm_features):
            return

        # Use clustering system for identity assignment
        cluster_id, distance, metadata = self.clustering_system.assign_identity(
            norm_features.flatten())

        # Convert cluster_id to person_id
        if cluster_id not in self.cluster_colors:
            person_id = f"Person_{self.next_person_id}"
            self._assign_cluster_color(cluster_id)
            self.next_person_id += 1
            self.stats['new_persons'] += 1
            detection.status = "new_person_created"
            logger.info(
                f"Created new person {person_id} with cluster {cluster_id}")
        else:
            person_id = self._get_person_id_from_cluster(cluster_id)
            detection.status = "person_reidentified"
            self.stats['reidentified_persons'] += 1
            logger.info(
                f"Re-identified as {person_id} with cluster {cluster_id}")

        detection.person_id = person_id
        track_state.person_id = person_id
        detection.similarity_score = 1.0 - distance

        # SOLUCIÓN: Inmediatamente después de asignar ID, agregar al cluster
        # para que empiece a crecer desde el primer frame
        if track_state.person_id:
            self._add_to_cluster(norm_features, track_state.person_id)
            track_state.cluster_contributions += 1
            self.stats['cluster_contributions'] += 1
            detection.status += "_and_added_to_cluster"

    def _calculate_similarity(self, features: np.ndarray, cluster) -> float:
        """Calculate similarity between features and cluster."""
        try:
            # Use Mahalanobis distance from cluster
            diff = features.flatten() - cluster.mu
            sigma_inv = np.linalg.inv(cluster.sigma)
            mahalanobis_dist = np.sqrt(diff.T @ sigma_inv @ diff)
            # Convert to similarity (0-1 scale) - más permisivo
            return max(0, 1.0 - mahalanobis_dist / 20.0)
        except:
            # Fallback to cosine similarity - más permisivo
            cosine_sim = float(np.dot(features.flatten(), cluster.mu) /
                               (np.linalg.norm(features) * np.linalg.norm(cluster.mu)))
            return max(0, cosine_sim + 0.2)  # Boost para ser más permisivo

    def _outlier_guard_check(self, features: np.ndarray, cluster) -> bool:
        """Check if features are outliers using statistical test."""
        try:
            diff = features.flatten() - cluster.mu
            sigma_inv = np.linalg.inv(cluster.sigma)
            mahalanobis_dist = np.sqrt(diff.T @ sigma_inv @ diff)
            return mahalanobis_dist > self.config['outlier_sigma_threshold']
        except:
            return False

    def _add_to_cluster(self, features: np.ndarray, person_id: str):
        """Add features to the corresponding cluster."""
        cluster_id = self._get_cluster_id_from_person(person_id)
        if cluster_id:
            # Update cluster through clustering system
            self.clustering_system.cluster_manager._update_cluster_incremental(
                self.clustering_system.cluster_manager.clusters[cluster_id],
                features.flatten()
            )

    def _check_cluster_size_and_purge(self, person_id: str) -> bool:
        """Check cluster size and purge if necessary."""
        cluster_id = self._get_cluster_id_from_person(person_id)
        if not cluster_id:
            return False

        cluster = self.clustering_system.get_cluster_by_id(cluster_id)
        if not cluster:
            return False

        if cluster.n_samples > self.config['max_cluster_size']:
            # Implement intelligent purging - remove oldest samples from HNSW
            purge_count = int(
                self.config['max_cluster_size'] * self.config['purge_percentage'])
            self._purge_cluster_samples(cluster_id, purge_count)
            return True
        return False

    def _purge_cluster_samples(self, cluster_id: str, purge_count: int):
        """Purge oldest samples from cluster."""
        # This would involve removing entries from HNSW index
        # For now, we'll just log the purge
        logger.info(f"Purged {purge_count} samples from cluster {cluster_id}")

    def _check_reactivated_track(self, detection: PersonDetection, track_state: TrackState, features: np.ndarray) -> bool:
        """Check for track reactivation."""
        for lost_track_id, lost_state in list(self.lost_tracks.items()):
            if lost_state.person_id:
                cluster_id = self._get_cluster_id_from_person(
                    lost_state.person_id)
                if cluster_id:
                    cluster = self.clustering_system.get_cluster_by_id(
                        cluster_id)
                    if cluster:
                        similarity = self._calculate_similarity(
                            features, cluster)
                        if similarity >= self.config['sim_threshold_assign_existing']:
                            # Reactivate
                            track_state.person_id = lost_state.person_id
                            detection.person_id = lost_state.person_id
                            detection.status = "reid_reactivated"
                            del self.lost_tracks[lost_track_id]
                            self.stats['tracks_reactivated'] += 1
                            self.stats['reid_reactivations'] += 1
                            logger.info(
                                f"Reactivated {lost_state.person_id} for track {track_state.track_id}")
                            return True
        return False

    def _find_best_detection_match(self, detections: List[PersonDetection], track) -> Optional[PersonDetection]:
        """Find best detection match using IoU."""
        best_iou = -1
        best_match = None

        for det in detections:
            iou_score = self._compute_iou(det.bbox, track.bbox)
            if iou_score > best_iou:
                best_iou = iou_score
                best_match = det

        if best_match and best_iou > 0.3:
            best_match.iou_score = best_iou
            return best_match
        return None

    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bounding boxes."""
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

    def _assign_cluster_color(self, cluster_id: str):
        """Assign color to new cluster."""
        color_idx = len(self.cluster_colors) % len(self.color_palette)
        self.cluster_colors[cluster_id] = self.color_palette[color_idx]

    def _get_person_id_from_cluster(self, cluster_id: str) -> str:
        """Get person ID from cluster ID."""
        # For now, create a mapping - in production this would be more sophisticated
        for i, (cid, _) in enumerate(self.cluster_colors.items()):
            if cid == cluster_id:
                return f"Person_{i + 1}"
        return f"Person_{len(self.cluster_colors) + 1}"

    def _get_cluster_id_from_person(self, person_id: str) -> Optional[str]:
        """Get cluster ID from person ID."""
        person_num = int(person_id.split('_')[1])
        cluster_ids = list(self.cluster_colors.keys())
        if person_num <= len(cluster_ids):
            return cluster_ids[person_num - 1]
        return None

    def _generate_cluster_visualization(self) -> np.ndarray:
        """Generate real-time cluster visualization frame."""
        # Create a simple visualization using OpenCV directly
        viz_frame = np.zeros((600, 800, 3), dtype=np.uint8)

        # Get all cluster data
        clusters = self.clustering_system.cluster_manager.clusters

        if not clusters:
            # No clusters yet
            cv2.putText(viz_frame, f'Frame {self.frame_count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(viz_frame, 'No clusters yet', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return viz_frame

        # Title
        cv2.putText(viz_frame, f'Frame {self.frame_count} - Cluster Evolution', (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(viz_frame, f'Active Clusters: {len(clusters)}', (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Simple visualization - show cluster info in a list
        y_pos = 120
        for i, (cluster_id, cluster) in enumerate(clusters.items()):
            person_id = self._get_person_id_from_cluster(cluster_id)
            color = self.cluster_colors.get(cluster_id, (128, 128, 128))

            # Draw cluster info
            text = f'{person_id}: {cluster.n_samples} samples'
            cv2.putText(viz_frame, text, (80, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw colored circle to represent cluster
            cv2.circle(viz_frame, (50, y_pos - 10), 15, color, -1)
            cv2.circle(viz_frame, (50, y_pos - 10), 15, (255, 255, 255), 2)

            # Draw size bar
            bar_width = min(300, cluster.n_samples * 3)
            cv2.rectangle(viz_frame, (350, y_pos - 15),
                          (350 + bar_width, y_pos - 5), color, -1)
            cv2.rectangle(viz_frame, (350, y_pos - 15),
                          (650, y_pos - 5), (255, 255, 255), 1)

            y_pos += 40

        # Add statistics
        stats_y = 400
        cv2.putText(viz_frame, 'Statistics:', (50, stats_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        stats_info = [
            f'Cluster Contributions: {self.stats["cluster_contributions"]}',
            f'Confirmation Hits: {self.stats["confirmation_hits_achieved"]}',
            f'Outliers Detected: {self.stats["outliers_detected"]}',
            f'Cluster Purges: {self.stats["cluster_purges"]}',
            f'Tracks Lost: {self.stats["tracks_lost"]}',
            f'Tracks Reactivated: {self.stats["tracks_reactivated"]}'
        ]

        for i, stat in enumerate(stats_info):
            cv2.putText(viz_frame, stat, (80, stats_y + 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return viz_frame

    def _store_cluster_visualization_data(self):
        """Store cluster data for post-processing visualization."""
        clusters = self.clustering_system.cluster_manager.clusters
        for cluster_id, cluster in clusters.items():
            viz_data = ClusterVisualizationData(
                frame_number=self.frame_count,
                cluster_id=cluster_id,
                centroid=cluster.mu.copy(),
                samples=[],  # Would store actual samples in production
                size=cluster.n_samples
            )
            self.cluster_viz_data.append(viz_data)

    def _annotate_frame(self, frame: np.ndarray, detections: List[PersonDetection]) -> np.ndarray:
        """Annotate frame with enhanced information."""
        for d in detections:
            x1, y1, x2, y2 = d.bbox

            # Get color from cluster
            color = (128, 128, 128)  # Default gray
            if d.person_id:
                cluster_id = self._get_cluster_id_from_person(d.person_id)
                if cluster_id and cluster_id in self.cluster_colors:
                    color = self.cluster_colors[cluster_id]

            # Create detailed label
            if d.person_id:
                label = f"{d.person_id}"
                if hasattr(d, 'similarity_score') and d.similarity_score > 0:
                    label += f" ({d.similarity_score:.2f})"
                label += f"\n[{d.status}]"
            else:
                label = f"Track_{d.track_id}\n[{d.status}]"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with background
            label_lines = label.split('\n')
            line_height = 15
            total_height = len(label_lines) * line_height + 5

            cv2.rectangle(frame, (x1, y1 - total_height),
                          (x1 + 200, y1), color, -1)

            for i, line in enumerate(label_lines):
                y_pos = y1 - total_height + (i + 1) * line_height
                cv2.putText(frame, line, (x1 + 2, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Add comprehensive frame info
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Active Clusters: {len(self.clustering_system.cluster_manager.clusters)}",
            f"Lost Tracks: {len(self.lost_tracks)}",
            f"Cluster Contributions: {self.stats['cluster_contributions']}",
            f"Outliers: {self.stats['outliers_detected']}",
            f"Purges: {self.stats['cluster_purges']}"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def _save_cluster_evolution_plot(self):
        """Save a comprehensive cluster evolution plot."""
        if not self.cluster_viz_data:
            return

        try:
            # Create evolution plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Plot 1: Cluster sizes over time
            cluster_sizes = defaultdict(list)
            frame_numbers = []

            for data in self.cluster_viz_data:
                if data.frame_number not in frame_numbers:
                    frame_numbers.append(data.frame_number)
                cluster_sizes[data.cluster_id].append(
                    (data.frame_number, data.size))

            for cluster_id, size_data in cluster_sizes.items():
                frames, sizes = zip(*size_data)
                color = np.array(self.cluster_colors.get(
                    cluster_id, (128, 128, 128))) / 255.0
                person_id = self._get_person_id_from_cluster(cluster_id)
                ax1.plot(frames, sizes, 'o-', color=color,
                         label=person_id, linewidth=2, markersize=4)

            ax1.set_xlabel('Frame Number')
            ax1.set_ylabel('Cluster Size (samples)')
            ax1.set_title('Cluster Growth Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Statistics over time
            stats_frames = list(
                range(0, self.frame_count, max(1, self.frame_count // 50)))
            ax2.plot(stats_frames, [self.stats['cluster_contributions']] * len(stats_frames),
                     'g-', label='Cluster Contributions', linewidth=2)
            ax2.plot(stats_frames, [self.stats['outliers_detected']] * len(stats_frames),
                     'r-', label='Outliers Detected', linewidth=2)
            ax2.plot(stats_frames, [self.stats['cluster_purges']] * len(stats_frames),
                     'b-', label='Cluster Purges', linewidth=2)

            ax2.set_xlabel('Frame Number')
            ax2.set_ylabel('Count')
            ax2.set_title('System Statistics')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = self.output_dir / 'cluster_evolution_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f'Cluster evolution plot saved to {plot_path}')
        except Exception as e:
            logger.warning(f'Could not save cluster evolution plot: {e}')

    def print_summary(self):
        """Print comprehensive summary with advanced statistics."""
        logger.info("=" * 80)
        logger.info("ADVANCED PERSON RE-ID PROCESSING SUMMARY")
        logger.info("=" * 80)

        # Basic Detection & Tracking
        logger.info("Detection & Tracking:")
        logger.info(f"  Total Detections: {self.stats['total_detections']}")
        logger.info(
            f"  High Confidence ReID: {self.stats['high_confidence_reid']}")

        # Clustering System
        logger.info("Advanced Clustering System:")
        logger.info(f"  Persons Created: {self.stats['new_persons']}")
        logger.info(
            f"  Persons Re-identified: {self.stats['reidentified_persons']}")
        logger.info(
            f"  Active Clusters: {len(self.clustering_system.cluster_manager.clusters)}")
        logger.info(
            f"  Cluster Contributions: {self.stats['cluster_contributions']}")
        logger.info(
            f"  Confirmation Hits Achieved: {self.stats['confirmation_hits_achieved']}")

        # Quality Control
        logger.info("Quality Control:")
        logger.info(f"  Outliers Detected: {self.stats['outliers_detected']}")
        logger.info(f"  Cluster Purges: {self.stats['cluster_purges']}")

        # Track Management
        logger.info("Track Management:")
        logger.info(f"  Tracks Lost: {self.stats['tracks_lost']}")
        logger.info(
            f"  Tracks Reactivated: {self.stats['tracks_reactivated']}")
        logger.info(
            f"  ReID Reactivations: {self.stats['reid_reactivations']}")

        # Cluster Details
        logger.info("Cluster Details:")
        for cluster_id, cluster in self.clustering_system.cluster_manager.clusters.items():
            person_id = self._get_person_id_from_cluster(cluster_id)
            logger.info(
                f"  {person_id} (cluster {cluster_id[:8]}): {cluster.n_samples} samples")

        logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Advanced Person Re-ID with Dynamic Clustering and Visualization")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file.")
    parser.add_argument("--output", type=str, default="output/test_demo_new_logic",
                        help="Directory for output files.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to process.")
    args = parser.parse_args()

    processor = AdvancedPersonReIDProcessor(
        input_video=args.video, output_dir=args.output)
    processor.process_video(max_frames=args.max_frames)
