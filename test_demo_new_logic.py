#!/usr/bin/env python3
"""
Video Person Re-ID with Advanced Logic and Tracking
===================================================

This script implements a sophisticated person re-identification pipeline for video
streams, based on an advanced logic flowchart. It combines object tracking with
deep learning-based Re-ID to maintain stable identities for people in a video.

Key Features:
- YOLO-based person detection.
- Huffman Tracker for robust object tracking across frames.
- OSNet for deep Re-ID feature extraction.
- HNSW index for fast similarity search.
- Multi-threshold logic for creating, assigning, and updating person IDs.
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


class VideoPersonReIDProcessor:
    """Orchestrates the entire Re-ID process based on the new logic."""

    def __init__(self, input_video: str, output_dir: str):
        self.input_video = input_video
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load thresholds from flowchart
        self.config = {
            'bbox_conf_threshold': 0.5,
            'person_conf_threshold': 0.9,
            'sim_threshold_new_cluster': 0.6,
            'sim_threshold_assign_existing': 0.7,
            'sim_threshold_update_cluster': 0.85,
            'kalman_max_lost': 100,
        }

        # Load models and components
        self.yolo_model = YOLO('yolov8s.pt')
        self.reid_system = OSNetReID()
        self.tracker = ByteTracker(
            frame_rate=30,
            track_thresh=0.5,
            track_buffer=self.config['kalman_max_lost'],
            match_thresh=0.8,
            high_thresh=0.6,
            low_thresh=0.1
        )
        self.hnsw_index = HNSWIndex()

        # State management
        self.next_person_id = 1
        self.track_to_person_map: Dict[int, str] = {}
        # New: Store a representative feature vector for each person
        self.person_feature_clusters: Dict[str, np.ndarray] = {}
        self.person_feature_counts: Dict[str, int] = {}
        self.person_colors: Dict[str, Tuple[int, int, int]] = {}
        self.color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]

        self.stats = {
            'total_detections': 0,
            'high_confidence_reid': 0,
            'new_persons': 0,
            'reidentified_persons': 0,
            'tentative_matches': 0,
            'lost_tracks': 0,
            'byte_tracked': 0,
            'byte_tracked_updated': 0,
            'byte_tracked_ignored': 0,
            'cluster_updates': 0,
        }

    def process_video(self, max_frames: Optional[int] = None):
        """Main processing loop for the video."""
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

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break

            # Step 1: Get all YOLO detections with their confidence
            all_detections = self._parse_yolo_results(
                self.yolo_model(frame, verbose=False), frame_count, frame)

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

            # Clean lost tracks from person mapping
            active_track_ids = {track.track_id for track in tracked_objects}
            lost_track_ids = []
            for track_id in list(self.track_to_person_map.keys()):
                if track_id not in active_track_ids:
                    lost_track_ids.append(track_id)
                    del self.track_to_person_map[track_id]
                    self.stats['lost_tracks'] += 1

            if lost_track_ids:
                logger.info(
                    f"Lost tracks: {lost_track_ids}, will reactivate ReID on next detection")

            # Step 3: Match tracked objects back to original detections
            processed_detections = []
            for track in tracked_objects:
                # Find the original detection that best matches this track
                best_iou = -1
                best_match = None
                for det in all_detections:
                    iou_score = self._compute_iou(det.bbox, track.bbox)
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_match = det

                if best_match and best_iou > 0.3:  # Lower threshold for ByteTracker
                    best_match.track_id = track.track_id

                    # Now apply Re-ID logic only if confidence is high enough
                    if best_match.confidence >= self.config['person_conf_threshold']:
                        self._apply_reid_logic(best_match, frame)

                    processed_detections.append(best_match)

            # Step 4: Annotate and save frame
            annotated_frame = self._annotate_frame(frame, processed_detections)
            writer.write(annotated_frame)

            logger.info(
                f"Frame {frame_count}: Detections={len(all_detections)}, Tracked={len(tracked_objects)}, Persons={self.next_person_id - 1}")
            frame_count += 1

        cap.release()
        writer.release()
        logger.info(
            f"Processing complete. Output video saved to {output_video_path}")
        self.print_summary()

    def _parse_yolo_results(self, yolo_results, frame_number, frame) -> List[PersonDetection]:
        """Extracts person detections from YOLO output."""
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

    def _apply_reid_logic(self, detection: PersonDetection, frame: np.ndarray):
        """The core Re-ID logic following the improved flowchart."""
        self.stats['high_confidence_reid'] += 1

        if detection.crop_image is None or detection.crop_image.size == 0:
            detection.status = "no_crop"
            return

        # Check if this track already has an assigned person ID (Kalman persistence)
        if detection.track_id in self.track_to_person_map:
            person_id = self.track_to_person_map[detection.track_id]
            detection.person_id = person_id
            detection.status = "kalman_tracked"

            # Extract features for cluster updating
            features = self.reid_system.extract_features(detection.crop_image)
            norm_features = features / np.linalg.norm(features)
            detection.reid_features = norm_features

            # Check if we should update the cluster (similarity ≥ 0.85)
            cluster_features = self.person_feature_clusters[person_id]
            similarity = np.dot(norm_features.flatten(),
                                cluster_features.flatten())

            if similarity >= self.config['sim_threshold_update_cluster']:
                self._update_person_cluster(person_id, norm_features)
                detection.status = "byte_tracked_updated"
                self.stats['byte_tracked_updated'] += 1
            else:
                detection.status = "byte_tracked_ignored"
                self.stats['byte_tracked_ignored'] += 1

            self.stats['byte_tracked'] += 1

            return

        # No ID assigned yet - need to do ReID
        features = self.reid_system.extract_features(detection.crop_image)
        norm_features = features / np.linalg.norm(features)
        detection.reid_features = norm_features

        self._identify_new_detection(detection)

    def _identify_new_detection(self, detection: PersonDetection):
        """Matches a new detection against all known person clusters following flowchart."""
        if not self.person_feature_clusters:
            self._create_new_person(detection)
            return

        best_match_id = None
        best_similarity = -1

        # Find the best match among all person clusters
        for person_id, cluster_features in self.person_feature_clusters.items():
            similarity = np.dot(
                detection.reid_features.flatten(), cluster_features.flatten())
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id

        # Following flowchart logic
        if best_similarity < self.config['sim_threshold_new_cluster']:
            # Similitud < 0.5: Crear nuevo clúster e ID
            self._create_new_person(detection)
        elif best_similarity >= self.config['sim_threshold_assign_existing']:
            # Similitud ≥ 0.7: Asignar ID existente
            detection.person_id = best_match_id
            detection.status = "re-identified"
            self.track_to_person_map[detection.track_id] = best_match_id
            self.stats['reidentified_persons'] += 1
            # Marcar como identificado - Kalman tomará seguimiento en próximos frames
            self._update_person_cluster(best_match_id, detection.reid_features)
        else:
            # Similitud entre 0.5-0.7: Descartar o reevaluar en siguiente frame
            detection.status = 'ignored_reevaluate'
            self.stats['tentative_matches'] += 1
            return  # no action

    def _create_new_person(self, detection: PersonDetection):
        """Creates a new person and initializes their feature cluster following flowchart."""
        person_id = f"Person_{self.next_person_id}"

        detection.person_id = person_id
        detection.status = "new_cluster_created"
        # Marcar como identificado - Kalman tomará seguimiento en próximos frames
        self.track_to_person_map[detection.track_id] = person_id

        # Initialize the feature cluster for the new person
        self.person_feature_clusters[person_id] = detection.reid_features.copy(
        )
        self.person_feature_counts[person_id] = 1

        # Assign a new color
        self.person_colors[person_id] = self.color_palette[self.next_person_id % len(
            self.color_palette)]

        self.stats['new_persons'] += 1
        self.next_person_id += 1

        logger.info(
            f"Created new person {person_id} with track_id {detection.track_id}")

    def _update_person_cluster(self, person_id: str, new_features: np.ndarray):
        """Updates a person's feature cluster using a running average following flowchart."""
        current_features = self.person_feature_clusters[person_id]
        current_count = self.person_feature_counts[person_id]

        # Update with a simple running average
        updated_features = (current_features * current_count +
                            new_features) / (current_count + 1)
        # Renormalize to keep it a unit vector
        self.person_feature_clusters[person_id] = updated_features / \
            np.linalg.norm(updated_features)
        self.person_feature_counts[person_id] += 1

        # Update statistics
        self.stats['cluster_updates'] += 1

        # Log cluster updates for debugging
        if self.person_feature_counts[person_id] % 10 == 0:
            logger.info(
                f"Updated cluster for {person_id}, count: {self.person_feature_counts[person_id]}")

    def _annotate_frame(self, frame: np.ndarray, detections: List[PersonDetection]) -> np.ndarray:
        """Draws bounding boxes and labels on the frame."""
        for d in detections:
            x1, y1, x2, y2 = d.bbox

            color = (128, 128, 128)  # Default gray for tentative/unknown
            if d.person_id and d.person_id in self.person_colors:
                color = self.person_colors[d.person_id]

            label = f"{d.person_id or f'Track_{d.track_id}'} [{d.status}]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Add a filled background for the label for better visibility
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
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
        """Prints a final summary of the processing statistics."""
        logger.info("=" * 50)
        logger.info("Processing Summary")
        logger.info("=" * 50)
        for key, value in self.stats.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
        logger.info("=" * 50)
        self._save_cluster_plot()

    def _save_cluster_plot(self):
        if len(self.person_feature_clusters) < 2:
            return  # not enough clusters

        # Ensure vectors are 2D (flatten if needed)
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
        plt.figure(figsize=(6, 6))
        for idx, (x, y) in enumerate(coords):
            pid = ids[idx]
            color = np.array(
                self.person_colors[pid])/255.0 if pid in self.person_colors else (0.5, 0.5, 0.5)
            plt.scatter(x, y, color=color, label=pid)
            plt.text(x+0.02, y+0.02, pid)
        plt.title('Cluster centres (PCA)')
        plt.legend()
        plot_path = self.output_dir / 'cluster_plot.png'
        plt.savefig(plot_path)
        plt.close()
        logger.info(f'Cluster plot saved to {plot_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Advanced Person Re-ID Processor")
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
