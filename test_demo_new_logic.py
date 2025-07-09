#!/usr/bin/env python3
"""
Video Person Re-ID with Advanced Logic and Tracking
===================================================

This script implements a sophisticated person re-identification pipeline for video
streams, based on an advanced logic flowchart. It combines object tracking with
deep learning-based Re-ID to maintain stable identities for people in a video.

Key Features:
- YOLO-based person detection.
- ByteTracker for robust object tracking across frames.
- OSNet for deep Re-ID feature extraction.
- HNSW index for fast similarity search.
- Multi-threshold logic for creating, assigning, and updating person IDs.
- Persistent vector space IDs with cluster-based tracking.
- Detailed statistics and annotated video output.
"""
from config.global_config import get_config, apply_preset
from src.index.hnsw_index import HNSWIndex
from src.reid.osnet_reid import OSNetReID
from src.tracking.byte_tracker import ByteTracker, Detection
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

# Add project root to path to allow direct imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Logging Setup ---
log_dir = Path("output/test_demo_new_logic/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "reid_new_logic.log"),
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


@dataclass
class TrackState:
    """Tracks the state of a ByteTracker track for ReID purposes."""
    track_id: int
    person_id: Optional[str] = None
    frames_lost: int = 0
    last_seen_frame: int = 0
    similarity_history: deque = None
    pending_cluster_update: bool = False

    def __post_init__(self):
        if self.similarity_history is None:
            self.similarity_history = deque(
                maxlen=5)  # Keep last 5 similarities


class VideoPersonReIDProcessor:
    """Orchestrates the entire Re-ID process based on the new flowchart logic."""

    def __init__(self, input_video: str, output_dir: str):
        self.input_video = input_video
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load thresholds from new flowchart
        self.config = {
            'bbox_conf_threshold': 0.5,
            'person_conf_threshold': 0.85, # <<--ATENTO AQUI
            'sim_threshold_new_cluster': 0.5,  # Updated to 0.5 as per flowchart
            'sim_threshold_assign_existing': 0.7,
            'sim_threshold_update_cluster': 0.85,
            'max_frames_lost': 100,  # N frames before reactivating ReID
            'similarity_frames_required': 2,  # Frames needed for cluster update
        }

        # Load models and components
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
        self.hnsw_index = HNSWIndex()

        # Enhanced state management for new flowchart
        self.next_person_id = 1
        self.track_states: Dict[int, TrackState] = {}

        # Vector space cluster management - this is the persistent ID space
        self.person_feature_clusters: Dict[str, np.ndarray] = {}
        self.person_feature_counts: Dict[str, int] = {}
        self.person_colors: Dict[str, Tuple[int, int, int]] = {}

        # Lost track management for ReID reactivation
        self.lost_tracks: Dict[int, TrackState] = {}
        self.frame_count = 0

        self.color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (165, 42, 42), (0, 128, 128), (128, 128, 0)
        ]

        self.stats = {
            'total_detections': 0,
            'high_confidence_reid': 0,
            'new_persons': 0,
            'reidentified_persons': 0,
            'tentative_matches': 0,
            'tracks_lost': 0,
            'tracks_reactivated': 0,
            'byte_tracked': 0,
            'byte_tracked_updated': 0,
            'byte_tracked_ignored': 0,
            'cluster_updates': 0,
            'reid_reactivations': 0,
        }

    def process_video(self, max_frames: Optional[int] = None):
        """Main processing loop for the video following the new flowchart."""
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {self.input_video}")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_video_path = self.output_dir / "video_output_new_logic.mp4"
        writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(
            *'mp4v'), fps, (frame_w, frame_h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and self.frame_count >= max_frames):
                break

            # Step 1: YOLO detecta persona
            all_detections = self._parse_yolo_results(
                self.yolo_model(frame, verbose=False), self.frame_count, frame)

            # Step 2: Convert to ByteTracker format and update tracker
            byte_detections = []
            for det in all_detections:
                byte_det = Detection(
                    bbox=np.array(det.bbox, dtype=np.float32),
                    score=det.confidence,
                    class_id=0
                )
                byte_detections.append(byte_det)

            # Update ByteTracker
            tracked_objects = self.tracker.update(byte_detections)

            # Step 3: Update track states and handle lost tracks
            self._update_track_states(tracked_objects)

            # Step 4: Match tracked objects back to original detections and apply flowchart
            processed_detections = []
            for track in tracked_objects:
                # Find the original detection that best matches this track
                best_match = self._find_best_detection_match(
                    all_detections, track)

                if best_match:
                    best_match.track_id = track.track_id

                    # Apply the new flowchart logic
                    self._apply_flowchart_logic(best_match, frame)
                    processed_detections.append(best_match)

            # Step 5: Annotate and save frame
            annotated_frame = self._annotate_frame(frame, processed_detections)
            writer.write(annotated_frame)

            logger.info(
                f"Frame {self.frame_count}: Detections={len(all_detections)}, "
                f"Tracked={len(tracked_objects)}, Persons={self.next_person_id - 1}")
            self.frame_count += 1

        cap.release()
        writer.release()
        logger.info(
            f"Processing complete. Output video saved to {output_video_path}")
        self.print_summary()

    def _parse_yolo_results(self, yolo_results, frame_number, frame) -> List[PersonDetection]:
        """Extracts person detections from YOLO output - Step: bbox_conf ≥ 0.5?"""
        detections = []
        for result in yolo_results:
            person_boxes = result.boxes[result.boxes.cls == 0]
            for box in person_boxes:
                confidence = box.conf.item()
                # Flowchart: bbox_conf ≥ 0.5?
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

    def _update_track_states(self, tracked_objects):
        """Update track states and handle lost tracks following flowchart."""
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

        # Handle lost tracks - Flowchart: Kalman pierde rastro ≥ N?
        newly_lost = []
        for track_id, track_state in list(self.track_states.items()):
            if track_id not in active_track_ids:
                track_state.frames_lost = self.frame_count - track_state.last_seen_frame

                if track_state.frames_lost >= self.config['max_frames_lost']:
                    # Move to lost tracks for potential ReID reactivation
                    self.lost_tracks[track_id] = track_state
                    del self.track_states[track_id]
                    newly_lost.append(track_id)
                    self.stats['tracks_lost'] += 1

        if newly_lost:
            logger.info(
                f"Lost tracks (≥{self.config['max_frames_lost']} frames): {newly_lost}")

    def _find_best_detection_match(self, detections: List[PersonDetection], track) -> Optional[PersonDetection]:
        """Find the best detection match for a track using IoU."""
        best_iou = -1
        best_match = None

        for det in detections:
            iou_score = self._compute_iou(det.bbox, track.bbox)
            if iou_score > best_iou:
                best_iou = iou_score
                best_match = det

        return best_match if best_iou > 0.3 else None

    def _apply_flowchart_logic(self, detection: PersonDetection, frame: np.ndarray):
        """Apply the complete flowchart logic for person identification."""
        track_state = self.track_states.get(detection.track_id)
        if not track_state:
            return

        # Flowchart: ¿Ya tiene ID asignado?
        if track_state.person_id is not None:
            # YES - Track ya identificado
            detection.person_id = track_state.person_id
            detection.status = "tracking_continued"
            self.stats['byte_tracked'] += 1

            # Flowchart: Continuar seguimiento + actualizar embeddings
            self._handle_identified_track(detection, track_state, frame)
        else:
            # NO - Nueva identificación needed
            # Flowchart: confidence_person ≥ 0.9?
            if detection.confidence >= self.config['person_conf_threshold']:
                self.stats['high_confidence_reid'] += 1
                # Flowchart: Extraer embedding + buscar en HNSW
                self._handle_new_identification(detection, track_state, frame)
            else:
                # Flowchart: Descartar bbox
                detection.status = "low_confidence_discarded"

    def _handle_identified_track(self, detection: PersonDetection, track_state: TrackState, frame: np.ndarray):
        """Handle already identified tracks - update embeddings if needed."""
        if detection.crop_image is None or detection.crop_image.size == 0:
            detection.status = "no_crop_available"
            return

        # Extract features for potential cluster updating
        features = self.reid_system.extract_features(detection.crop_image)
        norm_features = features / np.linalg.norm(features)
        detection.reid_features = norm_features

        # Get cluster features for similarity check
        person_id = track_state.person_id
        if person_id not in self.person_feature_clusters:
            return

        cluster_features = self.person_feature_clusters[person_id]
        similarity = np.dot(norm_features.flatten(),
                            cluster_features.flatten())

        # Store similarity history
        track_state.similarity_history.append(similarity)

        # Flowchart: Similitud ≥ 0.85 en ≥2 frames?
        if (similarity >= self.config['sim_threshold_update_cluster'] and
                len(track_state.similarity_history) >= self.config['similarity_frames_required']):

            # Check if we have enough high-similarity frames
            high_sim_count = sum(1 for s in track_state.similarity_history
                                 if s >= self.config['sim_threshold_update_cluster'])

            if high_sim_count >= self.config['similarity_frames_required']:
                # Flowchart: Agregar al clúster
                self._update_person_cluster(person_id, norm_features)
                detection.status = "cluster_updated"
                self.stats['byte_tracked_updated'] += 1
                self.stats['cluster_updates'] += 1
            else:
                detection.status = "similarity_insufficient"
                self.stats['byte_tracked_ignored'] += 1
        else:
            # Flowchart: Ignorar
            detection.status = "similarity_low"
            self.stats['byte_tracked_ignored'] += 1

    def _handle_new_identification(self, detection: PersonDetection, track_state: TrackState, frame: np.ndarray):
        """Handle new identification following flowchart logic."""
        if detection.crop_image is None or detection.crop_image.size == 0:
            detection.status = "no_crop_available"
            return

        # Extract embedding
        features = self.reid_system.extract_features(detection.crop_image)
        norm_features = features / np.linalg.norm(features)
        detection.reid_features = norm_features

        # Check if this might be a reactivated track
        if self._check_reactivated_track(detection, track_state):
            return

        # Search in existing clusters
        if not self.person_feature_clusters:
            # No existing clusters - create new
            self._create_new_person(detection, track_state)
            return

        # Find best match in vector space
        best_match_id = None
        best_similarity = -1

        for person_id, cluster_features in self.person_feature_clusters.items():
            similarity = np.dot(norm_features.flatten(),
                                cluster_features.flatten())
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id

        # Flowchart decision logic
        if best_similarity < self.config['sim_threshold_new_cluster']:
            # Flowchart: Similitud < 0.5 → Crear clúster e ID
            self._create_new_person(detection, track_state)
        elif best_similarity >= self.config['sim_threshold_assign_existing']:
            # Flowchart: Similitud ≥ 0.7 → Asignar ID existente
            self._assign_existing_person(detection, track_state, best_match_id)
        else:
            # Flowchart: Similitud 0.5-0.7 → Reevaluar
            detection.status = "reevaluate_next_frame"
            self.stats['tentative_matches'] += 1

    def _check_reactivated_track(self, detection: PersonDetection, track_state: TrackState) -> bool:
        """Check if this is a reactivated track from lost tracks."""
        # Look for similar tracks in lost_tracks that might be reactivated
        for lost_track_id, lost_state in list(self.lost_tracks.items()):
            if lost_state.person_id and lost_state.person_id in self.person_feature_clusters:
                cluster_features = self.person_feature_clusters[lost_state.person_id]
                similarity = np.dot(
                    detection.reid_features.flatten(), cluster_features.flatten())

                if similarity >= self.config['sim_threshold_assign_existing']:
                    # Reactivate this person ID
                    track_state.person_id = lost_state.person_id
                    detection.person_id = lost_state.person_id
                    detection.status = "reid_reactivated"

                    # Remove from lost tracks
                    del self.lost_tracks[lost_track_id]

                    self.stats['tracks_reactivated'] += 1
                    self.stats['reid_reactivations'] += 1

                    logger.info(
                        f"Reactivated person {lost_state.person_id} for track {track_state.track_id}")
                    return True

        return False

    def _create_new_person(self, detection: PersonDetection, track_state: TrackState):
        """Create a new person in the vector space - persistent ID creation."""
        person_id = f"Person_{self.next_person_id}"

        # Assign to both detection and track state
        detection.person_id = person_id
        track_state.person_id = person_id
        detection.status = "new_person_created"

        # Initialize the feature cluster in vector space
        self.person_feature_clusters[person_id] = detection.reid_features.copy(
        )
        self.person_feature_counts[person_id] = 1

        # Assign a color for visualization
        color_idx = (self.next_person_id - 1) % len(self.color_palette)
        self.person_colors[person_id] = self.color_palette[color_idx]

        self.stats['new_persons'] += 1
        self.next_person_id += 1

        logger.info(
            f"Created new person {person_id} for track {track_state.track_id}")

    def _assign_existing_person(self, detection: PersonDetection, track_state: TrackState, person_id: str):
        """Assign an existing person ID from the vector space."""
        detection.person_id = person_id
        track_state.person_id = person_id
        detection.status = "person_reidentified"

        # Update the cluster with this new observation
        self._update_person_cluster(person_id, detection.reid_features)

        self.stats['reidentified_persons'] += 1
        logger.info(
            f"Re-identified person {person_id} for track {track_state.track_id}")

    def _update_person_cluster(self, person_id: str, new_features: np.ndarray):
        """Update a person's feature cluster in the vector space using running average."""
        if person_id not in self.person_feature_clusters:
            return

        current_features = self.person_feature_clusters[person_id]
        current_count = self.person_feature_counts[person_id]

        # Running average update
        updated_features = (current_features * current_count +
                            new_features) / (current_count + 1)

        # Renormalize to maintain unit vector
        self.person_feature_clusters[person_id] = updated_features / \
            np.linalg.norm(updated_features)
        self.person_feature_counts[person_id] += 1

        # Log periodic updates
        if self.person_feature_counts[person_id] % 10 == 0:
            logger.info(
                f"Updated cluster for {person_id}, count: {self.person_feature_counts[person_id]}")

    def _annotate_frame(self, frame: np.ndarray, detections: List[PersonDetection]) -> np.ndarray:
        """Draw bounding boxes and labels with persistent vector space IDs."""
        for d in detections:
            x1, y1, x2, y2 = d.bbox

            # Use color from vector space if available
            color = (128, 128, 128)  # Default gray
            if d.person_id and d.person_id in self.person_colors:
                color = self.person_colors[d.person_id]

            # Create informative label with vector space ID
            if d.person_id:
                label = f"{d.person_id} [{d.status}]"
            else:
                label = f"Track_{d.track_id} [{d.status}]"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label background for better visibility
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add frame info
        info_text = f"Frame: {self.frame_count} | Active Persons: {len(self.person_feature_clusters)} | Lost Tracks: {len(self.lost_tracks)}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bounding boxes"""
        x1, y1, x2, y2 = bbox1
        x1_p, y1_p, x2_p, y2_p = bbox2

        # Intersection area
        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Union area
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def print_summary(self):
        """Print comprehensive summary of processing with vector space statistics."""
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY - New Flowchart Implementation")
        logger.info("=" * 60)

        # Basic statistics
        logger.info("Detection & Tracking:")
        logger.info(f"  Total Detections: {self.stats['total_detections']}")
        logger.info(
            f"  High Confidence ReID: {self.stats['high_confidence_reid']}")
        logger.info(f"  ByteTracker Active: {self.stats['byte_tracked']}")

        # Vector space statistics
        logger.info("Vector Space (Persistent IDs):")
        logger.info(f"  Persons Created: {self.stats['new_persons']}")
        logger.info(
            f"  Persons Re-identified: {self.stats['reidentified_persons']}")
        logger.info(
            f"  Active Person Clusters: {len(self.person_feature_clusters)}")
        logger.info(f"  Cluster Updates: {self.stats['cluster_updates']}")

        # Track management
        logger.info("Track Management:")
        logger.info(f"  Tracks Lost: {self.stats['tracks_lost']}")
        logger.info(
            f"  Tracks Reactivated: {self.stats['tracks_reactivated']}")
        logger.info(
            f"  ReID Reactivations: {self.stats['reid_reactivations']}")

        # Flowchart specific
        logger.info("Flowchart Logic:")
        logger.info(f"  Tentative Matches: {self.stats['tentative_matches']}")
        logger.info(f"  Cluster Updates: {self.stats['byte_tracked_updated']}")
        logger.info(f"  Updates Ignored: {self.stats['byte_tracked_ignored']}")

        # Cluster details
        logger.info("Cluster Details:")
        for person_id, count in self.person_feature_counts.items():
            logger.info(f"  {person_id}: {count} observations")

        logger.info("=" * 60)
        self._save_cluster_plot()

    def _save_cluster_plot(self):
        """Save a visualization of the vector space clusters."""
        if len(self.person_feature_clusters) < 2:
            return

        try:
            # Prepare data for PCA
            vectors = []
            for features in self.person_feature_clusters.values():
                if features.ndim > 1:
                    vectors.append(features.flatten())
                else:
                    vectors.append(features)
            vectors = np.stack(vectors)

            ids = list(self.person_feature_clusters.keys())
            pca = PCA(n_components=2)
            coords = pca.fit_transform(vectors)

            plt.figure(figsize=(8, 6))
            for idx, (x, y) in enumerate(coords):
                pid = ids[idx]
                color = np.array(
                    self.person_colors[pid])/255.0 if pid in self.person_colors else (0.5, 0.5, 0.5)
                plt.scatter(x, y, color=color, s=100,
                            label=f"{pid} ({self.person_feature_counts[pid]} obs)")
                plt.text(x+0.02, y+0.02, pid, fontsize=10)

            plt.title('Vector Space Clusters (PCA Projection)', fontsize=14)
            plt.xlabel('PC1', fontsize=12)
            plt.ylabel('PC2', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_path = self.output_dir / 'vector_space_clusters.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f'Vector space cluster plot saved to {plot_path}')
        except Exception as e:
            logger.warning(f'Could not save cluster plot: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Advanced Person Re-ID Processor with New Flowchart Logic")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file.")
    parser.add_argument("--output", type=str, default="output/test_demo_new_logic",
                        help="Directory for output files.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to process.")
    args = parser.parse_args()

    processor = VideoPersonReIDProcessor(
        input_video=args.video, output_dir=args.output)
    processor.process_video(max_frames=args.max_frames)
