#!/usr/bin/env python3
"""
Simplified POSE2ID Integration with Advanced Tracking
===================================================

This script integrates available POSE2ID components with the tracking logic
from test_demo_new_logic.py, using fallbacks when components are not available.
"""

from src.tracking.byte_tracker import ByteTracker, Detection
import os
import sys
import cv2
import time
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import tracking components

# Try to import POSE2ID components with fallbacks
try:
    from Pose2ID.NFC import NFC
except ImportError:
    print("Warning: Could not import NFC, using fallback")

    def NFC(feat_tensor, k1=3, k2=2):
        return feat_tensor  # Fallback implementation

# Try to import OSNet as fallback ReID
try:
    from src.reid.osnet_reid import OSNetReID
    REID_AVAILABLE = True
except ImportError:
    print("Warning: Could not import OSNetReID")
    REID_AVAILABLE = False

warnings.filterwarnings("ignore")

# --- Logging Setup ---
log_dir = Path("output/finalfinal_simple/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "pose2id_simple.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """Enhanced detection with POSE2ID features."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    frame_number: int
    track_id: Optional[int] = None
    person_id: Optional[str] = None
    status: str = "unprocessed"
    reid_features: Optional[np.ndarray] = None
    pose2id_features: Optional[np.ndarray] = None
    nfc_features: Optional[np.ndarray] = None
    crop_image: Optional[np.ndarray] = None
    similarity_score: float = 0.0


@dataclass
class TrackState:
    """Enhanced track state with POSE2ID integration."""
    track_id: int
    person_id: Optional[str] = None
    frames_lost: int = 0
    last_seen_frame: int = 0
    similarity_history: deque = None
    confirmation_hits: int = 0
    required_hits: int = 3
    pose2id_gallery: List[np.ndarray] = None
    nfc_enhanced_features: Optional[np.ndarray] = None
    generated_pose_count: int = 0

    def __post_init__(self):
        if self.similarity_history is None:
            self.similarity_history = deque(maxlen=10)
        if self.pose2id_gallery is None:
            self.pose2id_gallery = []


class SimplePOSE2IDProcessor:
    """Simplified POSE2ID processor with advanced tracking."""

    def __init__(self, input_video: str, output_dir: str):
        self.input_video = input_video
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.config = {
            'bbox_conf_threshold': 0.3,
            'person_conf_threshold': 0.7,
            'sim_threshold_new_cluster': 0.2,
            'sim_threshold_assign_existing': 0.4,
            'max_frames_lost': 30,
            'nfc_k1': 3,
            'nfc_k2': 2,
            'max_gallery_size': 50,
        }

        # Initialize YOLO and tracking
        self.yolo_model = YOLO('yolov8s.pt')
        self.tracker = ByteTracker(
            frame_rate=30,
            track_thresh=0.5,
            track_buffer=self.config['max_frames_lost'],
            match_thresh=0.8
        )

        # Initialize ReID system
        if REID_AVAILABLE:
            self.reid_system = OSNetReID()
            logger.info("Using OSNetReID for feature extraction")
        else:
            self.reid_system = None
            logger.warning("No ReID system available, using random features")

        # State management
        self.next_person_id = 1
        self.track_states: Dict[int, TrackState] = {}
        self.lost_tracks: Dict[int, TrackState] = {}
        self.frame_count = 0
        self.person_galleries: Dict[str, List[np.ndarray]] = {}

        # Statistics
        self.stats = {
            'total_detections': 0,
            'pose2id_extractions': 0,
            'nfc_enhancements': 0,
            'new_persons': 0,
            'reidentified_persons': 0,
        }

    def process_video(self, max_frames: Optional[int] = None):
        """Main processing loop with simplified POSE2ID integration."""
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {self.input_video}")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Setup video writer
        output_video_path = self.output_dir / "pose2id_simple_output.mp4"
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_w, frame_h)
        )

        logger.info(f"Processing video: {self.input_video}")
        logger.info(f"Output will be saved to: {output_video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and self.frame_count >= max_frames):
                break

            # Step 1: YOLO Detection
            detections = self._parse_yolo_results(
                self.yolo_model(frame, verbose=False), frame)

            # Step 2: ByteTracker update
            tracked_objects = self._update_tracker(detections)

            # Step 3: Update track states
            self._update_track_states(tracked_objects)

            # Step 4: Apply simplified POSE2ID pipeline
            processed_detections = self._apply_pose2id_pipeline(
                detections, tracked_objects, frame)

            # Step 5: Generate visualization
            annotated_frame = self._annotate_frame(frame, processed_detections)
            video_writer.write(annotated_frame)

            if self.frame_count % 100 == 0:
                logger.info(
                    f"Frame {self.frame_count}: Detections={len(detections)}, "
                    f"POSE2ID_extractions={self.stats['pose2id_extractions']}, "
                    f"Active_persons={len(self.person_galleries)}")

            self.frame_count += 1

        cap.release()
        video_writer.release()

        logger.info(f"Processing complete. Video saved to {output_video_path}")
        self.print_summary()

    def _parse_yolo_results(self, yolo_results, frame) -> List[PersonDetection]:
        """Parse YOLO results into PersonDetection objects."""
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
                        frame_number=self.frame_count,
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
        for track_id, track_state in list(self.track_states.items()):
            if track_id not in active_track_ids:
                track_state.frames_lost = self.frame_count - track_state.last_seen_frame
                if track_state.frames_lost >= self.config['max_frames_lost']:
                    self.lost_tracks[track_id] = track_state
                    del self.track_states[track_id]

    def _apply_pose2id_pipeline(self, detections: List[PersonDetection],
                                tracked_objects, frame: np.ndarray) -> List[PersonDetection]:
        """Apply simplified POSE2ID pipeline."""
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

            # Apply POSE2ID processing
            if best_match.crop_image is not None and best_match.crop_image.size > 0:
                # Step 1: Extract features (using available ReID system)
                pose2id_features = self._extract_pose2id_features(
                    best_match.crop_image)
                best_match.pose2id_features = pose2id_features
                self.stats['pose2id_extractions'] += 1

                # Step 2: Apply NFC if we have gallery
                if track_state.person_id and track_state.person_id in self.person_galleries:
                    nfc_features = self._apply_nfc_enhancement(
                        pose2id_features, track_state.person_id)
                    best_match.nfc_features = nfc_features
                    self.stats['nfc_enhancements'] += 1

                # Step 3: Identity assignment or verification
                if track_state.person_id is None:
                    # New identification
                    self._handle_new_identification(best_match, track_state)
                else:
                    # Continue tracking
                    best_match.person_id = track_state.person_id
                    best_match.status = "tracking_continued"

                    # Update gallery
                    self._update_person_gallery(
                        track_state.person_id, pose2id_features)

            processed_detections.append(best_match)

        return processed_detections

    def _extract_pose2id_features(self, crop_image: np.ndarray) -> np.ndarray:
        """Extract features using available ReID system."""
        if self.reid_system is not None:
            try:
                features = self.reid_system.extract_features(crop_image)
                return features / np.linalg.norm(features)  # Normalize
            except Exception as e:
                logger.warning(f"Error extracting features: {e}")
                return np.random.rand(512).astype(np.float32)
        else:
            # Fallback: simple color histogram features
            return self._extract_color_histogram(crop_image)

    def _extract_color_histogram(self, crop_image: np.ndarray) -> np.ndarray:
        """Extract simple color histogram features as fallback."""
        try:
            # Convert to HSV and compute histogram
            hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])

            # Concatenate and normalize
            features = np.concatenate(
                [hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            return features / np.linalg.norm(features)
        except:
            return np.random.rand(170).astype(np.float32)

    def _apply_nfc_enhancement(self, features: np.ndarray, person_id: str) -> np.ndarray:
        """Apply Neighbor Feature Centralization."""
        if person_id not in self.person_galleries or len(self.person_galleries[person_id]) < 2:
            return features

        try:
            # Get gallery features
            gallery_features = self.person_galleries[person_id]
            all_features = gallery_features + [features]

            # Convert to tensor
            feat_tensor = torch.stack([torch.from_numpy(f)
                                      for f in all_features])

            # Apply NFC
            enhanced_features = NFC(
                feat_tensor,
                k1=self.config['nfc_k1'],
                k2=self.config['nfc_k2']
            )

            # Return enhanced version of current features
            return enhanced_features[-1].numpy()
        except Exception as e:
            logger.warning(f"Error applying NFC: {e}")
            return features

    def _handle_new_identification(self, detection: PersonDetection, track_state: TrackState):
        """Handle new person identification."""
        if detection.pose2id_features is None:
            return

        # Check for reidentification
        best_match_id = None
        best_similarity = 0.0

        for person_id, gallery in self.person_galleries.items():
            if len(gallery) > 0:
                # Calculate similarity with gallery
                similarities = []
                for gallery_feat in gallery:
                    sim = self._calculate_cosine_similarity(
                        detection.pose2id_features, gallery_feat)
                    similarities.append(sim)

                avg_similarity = np.mean(similarities)
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_match_id = person_id

        # Decision making
        if best_similarity > self.config['sim_threshold_assign_existing']:
            # Reidentification
            detection.person_id = best_match_id
            track_state.person_id = best_match_id
            detection.status = "reidentified"
            detection.similarity_score = best_similarity
            self.stats['reidentified_persons'] += 1

            # Update gallery
            self._update_person_gallery(
                best_match_id, detection.pose2id_features)
        else:
            # New person
            person_id = f"Person_{self.next_person_id}"
            detection.person_id = person_id
            track_state.person_id = person_id
            detection.status = "new_person"
            self.next_person_id += 1
            self.stats['new_persons'] += 1

            # Initialize gallery
            self.person_galleries[person_id] = [detection.pose2id_features]

    def _update_person_gallery(self, person_id: str, features: np.ndarray):
        """Update person gallery with new features."""
        if person_id not in self.person_galleries:
            self.person_galleries[person_id] = []

        self.person_galleries[person_id].append(features)

        # Limit gallery size
        if len(self.person_galleries[person_id]) > self.config['max_gallery_size']:
            self.person_galleries[person_id].pop(0)

    def _calculate_cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors."""
        try:
            dot_product = np.dot(feat1, feat2)
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except:
            return 0.0

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

    def _annotate_frame(self, frame: np.ndarray, detections: List[PersonDetection]) -> np.ndarray:
        """Annotate frame with POSE2ID results."""
        for d in detections:
            x1, y1, x2, y2 = d.bbox

            # Color based on person ID
            if d.person_id:
                # Generate consistent color from person ID
                person_num = int(d.person_id.split('_')[1])
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                          (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
                color = colors[person_num % len(colors)]
            else:
                color = (128, 128, 128)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Create label
            if d.person_id:
                label = f"{d.person_id}"
                if d.similarity_score > 0:
                    label += f" ({d.similarity_score:.2f})"
                label += f"\n[{d.status}]"

                # Add POSE2ID info
                if d.pose2id_features is not None:
                    label += "\n[Features]"
                if d.nfc_features is not None:
                    label += "\n[NFC]"
            else:
                label = f"Track_{d.track_id}\n[{d.status}]"

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
            f"Active Persons: {len(self.person_galleries)}",
            f"Feature Extractions: {self.stats['pose2id_extractions']}",
            f"NFC Enhancements: {self.stats['nfc_enhancements']}",
            f"ReID System: {'OSNet' if REID_AVAILABLE else 'Color Histogram'}"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def print_summary(self):
        """Print comprehensive summary."""
        logger.info("=" * 80)
        logger.info("SIMPLIFIED POSE2ID PROCESSING SUMMARY")
        logger.info("=" * 80)

        logger.info("Detection & Tracking:")
        logger.info(f"  Total Detections: {self.stats['total_detections']}")
        logger.info(f"  New Persons: {self.stats['new_persons']}")
        logger.info(
            f"  Reidentified Persons: {self.stats['reidentified_persons']}")

        logger.info("Feature Processing:")
        logger.info(
            f"  Feature Extractions: {self.stats['pose2id_extractions']}")
        logger.info(f"  NFC Enhancements: {self.stats['nfc_enhancements']}")
        logger.info(
            f"  ReID System: {'OSNet' if REID_AVAILABLE else 'Color Histogram'}")

        logger.info("Gallery Statistics:")
        for person_id, gallery in self.person_galleries.items():
            logger.info(f"  {person_id}: {len(gallery)} features")

        logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simplified POSE2ID Integration with Advanced Tracking")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file.")
    parser.add_argument("--output", type=str, default="output/finalfinal_simple",
                        help="Directory for output files.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to process.")
    args = parser.parse_args()

    processor = SimplePOSE2IDProcessor(
        input_video=args.video,
        output_dir=args.output
    )
    processor.process_video(max_frames=args.max_frames)
