#!/usr/bin/env python3
"""
Video-based Person Re-identification with Gaussian Clustering
============================================================

This script processes video files to detect and re-identify persons using:
1. YOLO detection with dual-threshold approach (50% for display, 90% for clustering)
2. ReID feature extraction with OSNet
3. Gaussian clustering for person identity assignment
4. Real-time video output with bounding boxes and cluster-based colors
5. Comprehensive person counting and statistics

Based on test_cluster.py logic but adapted for video processing.
Uses three_person_dataset_90_confidence configuration for optimal performance.
"""

from src.clustering.gaussian_cluster_manager import GaussianClusterManager
from src.index.hnsw_index import HNSWIndex
from src.reid.osnet_reid import OSNetReID
from src.tracking.huffman_tracker import PersonTracker
from config.global_config import get_config, apply_preset, config
import os
import sys
import cv2
import time
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import torch
from ultralytics import YOLO

# Add src to path
sys.path.append('src')

# Import project modules


@dataclass
class PersonDetection:
    """Represents a person detection with all relevant information"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    frame_number: int
    is_high_confidence: bool
    crop_image: Optional[np.ndarray] = None
    reid_features: Optional[np.ndarray] = None
    person_id: Optional[str] = None
    similarity_score: Optional[float] = None
    track_id: Optional[int] = None
    status: str = "unprocessed"


class VideoPersonReIDProcessor:
    """
    Video-based Person Re-identification with Gaussian Clustering

    This processor applies the same logic as test_cluster.py but to video streams:
    - Detects persons with YOLO (50% threshold for display)
    - Validates high-confidence detections (90% threshold)
    - Extracts ReID features for validated detections
    - Applies Gaussian clustering for person identity assignment
    - Generates annotated video output with cluster-based colors
    - Provides comprehensive person counting and statistics
    """

    def __init__(self, input_video: str, output_dir: str, detection_threshold: float = 0.5, reid_threshold: float = 0.9):
        self.input_video = input_video
        self.output_dir = Path(output_dir)
        self.detection_threshold = detection_threshold
        self.reid_threshold = reid_threshold

        # New logic thresholds from the flowchart
        self.logic_config = {
            'person_conf_threshold': 0.9,
            'bbox_conf_threshold': 0.5,
            'reid_sim_threshold_new_id': 0.5,
            'reid_sim_threshold_existing': 0.7,
            'embedding_update_threshold': 0.85,
        }

        # Initialize components
        self.yolo_model = YOLO('yolov8s.pt')
        self.reid_system = OSNetReID()
        self.hnsw_index = HNSWIndex()
        self.tracker = PersonTracker()

        self.next_person_id = 1
        self.track_to_person_map = {}

        # Initialize the tracker
        self.tracker = PersonTracker(
            max_disappeared=get_config('tracking')['max_disappeared'],
            max_distance=get_config('tracking')['max_distance']
        )

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "crops").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Apply configuration preset
        apply_preset(config_preset)
        print(f"✅ Applied configuration preset: {config_preset}")

        # Get configurations
        self.detection_config = get_config('detection')
        self.reid_config = get_config('reid')
        self.clustering_config = {
            'similarity_threshold': 0.5,
            'expected_persons': get_config('gaussian_clustering')['expected_persons']
        }
        self.indexing_config = get_config('indexing')

        # Setup logging
        self.setup_logging()

        # Initialize models
        self.initialize_models()

        # Initialize clustering and ReID system
        self.initialize_reid_system()

        # Initialize color mapping system
        self.initialize_color_mapping()

        # Statistics tracking
        self._initialize_stats()
        self._initialize_color_mapping()

        # Person management
        self.person_galleries = defaultdict(list)
        self.person_confidences = defaultdict(list)
        self.cluster_to_person_map = {}
        self.next_person_id = 1
        self.person_stats = {}

        # External ID management for HNSW
        self.external_to_cluster_map = {}
        self.detection_counter = 0

        self.logger.info("✅ Video processor initialized successfully!")

    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_file = self.output_dir / "logs" / "video_reid_processor.log"

        # Create logger
        self.logger = logging.getLogger("VideoReIDProcessor")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def initialize_models(self):
        """Initialize YOLO and ReID models"""
        self.logger.info("🚀 Initializing detection and ReID models...")

        # Initialize YOLO model
        yolo_model_path = self.detection_config['model_path']
        self.yolo_model = YOLO(yolo_model_path)
        self.logger.info(f"✅ YOLO model loaded: {yolo_model_path}")

        # Initialize ReID model
        self.reid_model = OSNetReID()
        self.logger.info(
            f"✅ ReID model loaded: {self.reid_config['model_name']}")

    def initialize_reid_system(self):
        """Initialize the ReID system with HNSW index."""
        self.logger.info("🧠 Initializing ReID system...")

        self.hnsw_index = HNSWIndex(
            dim=self.reid_config['feature_dim'],
            max_elements=self.indexing_config['max_elements'],
            M=self.indexing_config['M'],
            ef_construction=self.indexing_config['ef_construction'],
            space=self.indexing_config['space']
        )
        self.logger.info("✅ ReID system initialized")

    def initialize_color_mapping(self):
        """Initialize color mapping system for cluster-based visualization"""
        # Define a comprehensive color palette for different clusters/persons
        self.color_palette = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 128),    # Purple
            (255, 165, 0),    # Orange
            (0, 128, 128),    # Teal
            (128, 128, 0),    # Olive
            (255, 192, 203),  # Pink
            (165, 42, 42),    # Brown
            (128, 128, 128),  # Gray
            (255, 20, 147),   # Deep Pink
            (0, 191, 255),    # Deep Sky Blue
            (50, 205, 50),    # Lime Green
            (255, 69, 0),     # Red Orange
            (138, 43, 226),   # Blue Violet
            (255, 215, 0),    # Gold
            (220, 20, 60)     # Crimson
        ]

        # Color mapping for different states
        self.state_colors = {
            'low_confidence': (0, 0, 255),      # Red for low confidence
            'processing': (0, 255, 255),        # Yellow for processing
            'unknown': (128, 128, 128),         # Gray for unknown
            'error': (255, 0, 255)              # Magenta for errors
        }

        # Maps to track color assignments
        self.cluster_color_map = {}
        self.person_color_map = {}

        self.logger.info("🎨 Color mapping system initialized")

    def get_cluster_color(self, cluster_id: str) -> Tuple[int, int, int]:
        """
        Get consistent color for a cluster ID

        Args:
            cluster_id: Cluster identifier

        Returns:
            BGR color tuple
        """
        if cluster_id not in self.cluster_color_map:
            # Assign color based on cluster index
            color_index = len(self.cluster_color_map) % len(self.color_palette)
            self.cluster_color_map[cluster_id] = self.color_palette[color_index]

        return self.cluster_color_map[cluster_id]

    def get_person_color(self, person_id: str) -> Tuple[int, int, int]:
        """
        Get consistent color for a person ID

        Args:
            person_id: Person identifier

        Returns:
            BGR color tuple
        """
        if person_id not in self.person_color_map:
            # Assign color based on person index
            color_index = len(self.person_color_map) % len(self.color_palette)
            self.person_color_map[person_id] = self.color_palette[color_index]

        return self.person_color_map[person_id]

    def _initialize_stats(self):
        self.stats = {
            'total_detections': 0, 'high_confidence_detections': 0, 'low_confidence_detections': 0,
            'persons_identified': 0, 'reid_processed': 0, 'tentative_assignments': 0,
            'embedding_updates': 0, 'new_persons_created': 0
        }

    def _initialize_color_mapping(self):
        """Initialize color mapping system for cluster-based visualization"""
        # Define a comprehensive color palette for different clusters/persons
        self.color_palette = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 128),    # Purple
            (255, 165, 0),    # Orange
            (0, 128, 128),    # Teal
            (128, 128, 0),    # Olive
            (255, 192, 203),  # Pink
            (165, 42, 42),    # Brown
            (128, 128, 128),  # Gray
            (255, 20, 147),   # Deep Pink
            (0, 191, 255),    # Deep Sky Blue
            (50, 205, 50),    # Lime Green
            (255, 69, 0),     # Red Orange
            (138, 43, 226),   # Blue Violet
            (255, 215, 0),    # Gold
            (220, 20, 60)     # Crimson
        ]

        # Color mapping for different states
        self.state_colors = {
            'low_confidence': (0, 0, 255),      # Red for low confidence
            'processing': (0, 255, 255),        # Yellow for processing
            'unknown': (128, 128, 128),         # Gray for unknown
            'error': (255, 0, 255)              # Magenta for errors
        }

        # Maps to track color assignments
        self.cluster_color_map = {}
        self.person_color_map = {}

        self.logger.info("🎨 Color mapping system initialized")

    def detect_persons_in_frame(self, frame: np.ndarray, frame_number: int) -> List[PersonDetection]:
        """
        Detect persons in frame using YOLO with dual-threshold approach

        Args:
            frame: Input frame
            frame_number: Current frame number

        Returns:
            List of PersonDetection objects
        """
        start_time = time.time()

        # Run YOLO detection
        results = self.yolo_model(frame, verbose=False)
        detections = []

        for result in results:
            if result.boxes is not None:
                # Filter for person class (class 0 in COCO)
                person_boxes = result.boxes[result.boxes.cls == 0]

                for box in person_boxes:
                    confidence = box.conf.item()

                    # Only include detections above detection threshold
                    if confidence >= self.detection_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Extract crop for high-confidence detections
                        crop_image = None
                        if confidence >= self.reid_threshold:
                            # Validate crop dimensions
                            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                                crop_image = frame[y1:y2, x1:x2]
                                if crop_image.size == 0:
                                    crop_image = None

                        detection = PersonDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            frame_number=frame_number,
                            is_high_confidence=confidence >= self.reid_threshold,
                            crop_image=crop_image
                        )
                        detections.append(detection)

        # Update statistics
        detection_time = time.time() - start_time
        self.stats['processing_times']['detection'].append(detection_time)
        self.stats['total_detections'] += len(detections)
        self.stats['high_confidence_detections'] += sum(
            1 for d in detections if d.is_high_confidence)
        self.stats['low_confidence_detections'] += sum(
            1 for d in detections if not d.is_high_confidence)

        return detections

    def _get_yolo_detections(self, frame: np.ndarray, frame_number: int) -> List[PersonDetection]:
        """
        Detect persons in frame using YOLO with dual-threshold approach

        Args:
            frame: Input frame
            frame_number: Current frame number

        Returns:
            List of PersonDetection objects
        """
        start_time = time.time()

        # Run YOLO detection
        results = self.yolo_model(frame, verbose=False)
        detections = []

        for result in results:
            if result.boxes is not None:
                # Filter for person class (class 0 in COCO)
                person_boxes = result.boxes[result.boxes.cls == 0]

                for box in person_boxes:
                    confidence = box.conf.item()

                    # Only include detections above detection threshold
                    if confidence >= self.detection_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Extract crop for high-confidence detections
                        crop_image = None
                        if confidence >= self.reid_threshold:
                            # Validate crop dimensions
                            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                                crop_image = frame[y1:y2, x1:x2]
                                if crop_image.size == 0:
                                    crop_image = None

                        detection = PersonDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            frame_number=frame_number,
                            is_high_confidence=confidence >= self.reid_threshold,
                            crop_image=crop_image
                        )
                        detections.append(detection)

        # Update statistics
        detection_time = time.time() - start_time
        self.stats['processing_times']['detection'].append(detection_time)
        self.stats['total_detections'] += len(detections)
        self.stats['high_confidence_detections'] += sum(
            1 for d in detections if d.is_high_confidence)
        self.stats['low_confidence_detections'] += sum(
            1 for d in detections if not d.is_high_confidence)

        return detections

    def _process_high_confidence_detections(self, frame_detections: List[PersonDetection]) -> List[PersonDetection]:
        if not frame_detections:
            return []

        for detection in frame_detections:
            try:
                if detection.crop_image is None:
                    continue

                detection.reid_features = self.reid_model.extract_features(
                    detection.crop_image)
                normalized_features = detection.reid_features / \
                    np.linalg.norm(detection.reid_features)

                if self.hnsw_index.index.get_current_count() == 0:
                    person_id = f"Person_{self.next_person_id}"
                    self.hnsw_index.add_vector(
                        normalized_features, external_id=person_id)
                    detection.person_id = person_id
                    self.next_person_id += 1
                else:
                    search_results = self.hnsw_index.search(
                        normalized_features, k=1)
                    if search_results and search_results['external_ids']:
                        distance = search_results['distances'][0]
                        similarity = 1 - distance
                        if similarity >= self.clustering_config['similarity_threshold']:
                            detection.person_id = search_results['external_ids'][0]
                        else:
                            person_id = f"Person_{self.next_person_id}"
                            self.hnsw_index.add_vector(
                                normalized_features, external_id=person_id)
                            detection.person_id = person_id
                            self.next_person_id += 1
                    else:
                        person_id = f"Person_{self.next_person_id}"
                        self.hnsw_index.add_vector(
                            normalized_features, external_id=person_id)
                        detection.person_id = person_id
                        self.next_person_id += 1

                if detection.person_id:
                    if detection.person_id not in self.person_stats:
                        self.person_stats[detection.person_id] = {
                            'crops_saved': 0, 'frame_numbers': [], 'confidences': []}
                    self.person_stats[detection.person_id]['crops_saved'] += 1
                    self.person_stats[detection.person_id]['frame_numbers'].append(
                        detection.frame_number)
                    self.person_stats[detection.person_id]['confidences'].append(
                        detection.confidence)

            except Exception as e:
                self.logger.error(
                    f"Failed to process high-confidence detection: {e}")

        return frame_detections

    def save_person_crop(self, crop: np.ndarray, person_id: str, frame_number: int):
        """Save person crop to disk"""
        person_dir = self.output_dir / "crops" / person_id
        person_dir.mkdir(exist_ok=True)

        crop_path = person_dir / f"frame_{frame_number:06d}.jpg"
        cv2.imwrite(str(crop_path), crop)

    def _annotate_frame(self, frame: np.ndarray, detections: List[PersonDetection]) -> np.ndarray:
        """
        Annotate frame with bounding boxes and person IDs using cluster-based colors

        Args:
            frame: Input frame
            detections: List of detections

        Returns:
            Annotated frame with cluster-based color coding
        """
        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            confidence = detection.confidence

            # Determine color and label based on cluster assignment and confidence
            if detection.is_high_confidence and detection.person_id and detection.person_id != "Unknown":
                # High confidence with successful cluster assignment - use cluster-based color
                color = self.get_person_color(detection.person_id)

                if detection.status == "tentative":
                    label = f"{detection.person_id}? (sim:{detection.similarity_score:.2f})"
                else:
                    label = f"{detection.person_id} ({confidence:.2f})"

                # Track person detection in this frame
                self.stats['person_detection_frames'][detection.person_id].add(
                    detection.frame_number)

            elif detection.is_high_confidence:
                # High confidence but processing failed or in progress
                color = self.state_colors['processing']
                label = f"Processing... ({confidence:.2f})"
            else:
                # Low confidence - show question mark
                color = self.state_colors['low_confidence']
                label = f"? ({confidence:.2f})"

            # Draw bounding box with cluster-based color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1),
                          color, -1)

            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Update unique persons count
        current_unique_persons = len(set(d.person_id for d in detections
                                         if d.person_id and d.person_id != "Unknown"))
        self.stats['unique_persons_detected'] = max(
            self.stats['unique_persons_detected'], current_unique_persons)

        # Add comprehensive system information
        info_text = [
            f"Frame: {self.stats['total_frames']}",
            f"Detections: {len(detections)}",
            f"High Conf: {sum(1 for d in detections if d.is_high_confidence)}",
            f"Persons: {self.stats['persons_identified']}",
            f"Unique: {self.stats['unique_persons_detected']}",
            f"Clusters: {self.stats['clusters_created']}",
            f"Detection: {self.detection_threshold:.0%} | ReID: {self.reid_threshold:.0%}"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(annotated_frame, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add color legend for identified persons
        if self.person_color_map:
            legend_y_start = 30 + len(info_text) * 25 + 20
            cv2.putText(annotated_frame, "Person Colors:", (10, legend_y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for i, (person_id, color) in enumerate(self.person_color_map.items()):
                legend_y = legend_y_start + 20 + i * 20
                # Draw color square
                cv2.rectangle(annotated_frame, (10, legend_y - 10),
                              (25, legend_y + 5), color, -1)
                # Draw person ID
                cv2.putText(annotated_frame, person_id, (30, legend_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return annotated_frame

    def process_video(self, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Process video with person detection and clustering

        Args:
            max_frames: Maximum frames to process (None for all)

        Returns:
            Processing statistics
        """
        self.logger.info(f"🎬 Starting video processing: {self.input_video}")

        # Open video
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_video}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        self.logger.info(
            f"📹 Video properties: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")

        # Setup video writer
        timestamp = int(time.time())
        output_video_path = self.output_dir / "videos" / \
            f"reid_clustering_output_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path),
                              fourcc, fps, (width, height))

        # Processing loop
        start_time = time.time()
        frame_count = 0

        try:
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret or (max_frames and frame_count >= max_frames):
                        break

                    frame_start_time = time.time()

                    # Step 1: Get all detections for the current frame
                    all_detections = self._get_yolo_detections(
                        frame, frame_count)

                    # Filter detections based on bbox_conf_threshold
                    valid_detections = [
                        d for d in all_detections if d.confidence >= self.logic_config['bbox_conf_threshold']]

                    # Prepare for tracker
                    tracker_input = [
                        {'bbox': d.bbox, 'confidence': d.confidence} for d in valid_detections]

                    # Step 2: Update tracker
                    tracked_objects = self.tracker.update(tracker_input)

                    processed_detections = []
                    # Map tracker outputs back to our PersonDetection objects
                    for track_id, track_data in tracked_objects.items():
                        # Find the original detection to get crop_image, etc.
                        original_detection = next(
                            (d for d in valid_detections if d.bbox == track_data['bbox']), None)
                        if original_detection:
                            original_detection.track_id = track_id

                            # Step 3: Apply the main Re-ID logic
                            if original_detection.confidence >= self.logic_config['person_conf_threshold']:
                                self._apply_reid_logic(original_detection)

                            processed_detections.append(original_detection)

                    # Stage 3: Annotate frame
                    annotated_frame = self._annotate_frame(
                        frame, processed_detections)

                    # Write frame
                    out.write(annotated_frame)

                    # Update statistics
                    frame_time = time.time() - frame_start_time
                    self.stats['processing_times']['frame'].append(frame_time)
                    self.stats['total_frames'] = frame_count + 1

                    # Update progress
                    pbar.update(1)

                    # Log progress periodically
                    if frame_count % 30 == 0:
                        current_fps = 1.0 / frame_time if frame_time > 0 else 0
                        high_conf_count = sum(
                            1 for d in processed_detections if d.is_high_confidence)
                        self.logger.info(f"Frame {frame_count}/{total_frames} | "
                                         f"Detections: {len(processed_detections)} | "
                                         f"High Conf: {high_conf_count} | "
                                         f"Persons: {self.stats['persons_identified']} | "
                                         f"FPS: {current_fps:.1f}")

                    frame_count += 1

        except KeyboardInterrupt:
            self.logger.info("⚠️ Processing interrupted by user")

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        # Calculate final statistics
        processing_time = time.time() - start_time
        avg_fps = frame_count / processing_time if processing_time > 0 else 0

        # Update final statistics
        self.stats.update({
            'processing_time': processing_time,
            'avg_fps': avg_fps,
            'output_video': str(output_video_path),
            'clusters_created': self.stats['clusters_created'],
            'final_persons': self.stats['persons_identified']
        })

        # Save comprehensive statistics
        self.save_statistics()

        self.logger.info(
            f"✅ Processing completed: {frame_count} frames in {processing_time:.2f}s")
        self.logger.info(f"📊 Average FPS: {avg_fps:.2f}")
        self.logger.info(
            f"👥 Persons identified: {self.stats['persons_identified']}")
        self.logger.info(f"🎥 Output video: {output_video_path}")

        return self.stats

    def save_statistics(self):
        """Save comprehensive processing statistics"""
        stats_path = self.output_dir / "data" / "processing_statistics.json"

        # Calculate cluster statistics
        cluster_stats = {}
        # for cluster_id, cluster in self.cluster_manager.clusters.items():
        #     cluster_stats[cluster_id] = {
        #         'n_samples': cluster.n_samples,
        #         'creation_time': cluster.creation_time,
        #         'last_update_time': cluster.last_update_time,
        #         'person_id': self.cluster_to_person_map.get(cluster_id, 'Unknown')
        #     }

        # Prepare statistics for JSON serialization
        json_stats = {
            'video_info': {
                'input_path': self.input_video,
                'output_path': self.stats['output_video'],
                'total_frames': self.stats['total_frames'],
                'processing_time': self.stats['processing_time'],
                'avg_fps': self.stats['avg_fps']
            },
            'detection_stats': {
                'total_detections': self.stats['total_detections'],
                'high_confidence_detections': self.stats['high_confidence_detections'],
                'low_confidence_detections': self.stats['low_confidence_detections'],
                'detection_threshold': self.detection_threshold,
                'reid_threshold': self.reid_threshold
            },
            'clustering_stats': {
                'clusters_created': self.stats['clusters_created'],
                'persons_identified': self.stats['persons_identified'],
                'unique_persons_detected': self.stats['unique_persons_detected'],
                'tentative_assignments': self.stats['tentative_assignments'],
                'reid_processed': self.stats['reid_processed'],
                'cluster_details': cluster_stats,
                'color_mapping': {
                    'cluster_colors': {k: list(v) for k, v in self.cluster_color_map.items()},
                    'person_colors': {k: list(v) for k, v in self.person_color_map.items()}
                }
            },
            'person_galleries': {
                person_id: len(crops) for person_id, crops in self.person_galleries.items()
            },
            'configuration': {
                'detection_threshold': self.detection_threshold,
                'reid_threshold': self.reid_threshold,
                'similarity_threshold_high': self.clustering_config['similarity_threshold'],
                # This line was not in the new_code, but should be consistent
                'similarity_threshold_low': self.clustering_config['similarity_threshold'],
                'expected_persons': self.clustering_config['expected_persons']
            },
            'processing_times': {
                stage: {
                    'mean': np.mean(times) if times else 0,
                    'std': np.std(times) if times else 0,
                    'min': np.min(times) if times else 0,
                    'max': np.max(times) if times else 0,
                    'total': np.sum(times) if times else 0
                }
                for stage, times in self.stats['processing_times'].items()
            }
        }

        with open(stats_path, 'w') as f:
            json.dump(json_stats, f, indent=2)

        self.logger.info(f"📊 Statistics saved to: {stats_path}")

    def get_person_gallery(self, person_id: str, max_crops: int = 20) -> List[np.ndarray]:
        """Get person gallery crops"""
        crops = self.person_galleries.get(person_id, [])
        return crops[:max_crops]

    def print_summary(self):
        """Print comprehensive processing summary with enhanced person statistics"""
        print("\n" + "="*80)
        print("VIDEO PERSON RE-ID WITH CLUSTERING - PROCESSING SUMMARY")
        print("="*80)

        print(f"📹 Video Input: {self.input_video}")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"🎥 Output Video: {self.stats['output_video']}")

        print(f"\n📊 Processing Statistics:")
        print(f"  • Total frames processed: {self.stats['total_frames']}")
        print(
            f"  • Processing time: {self.stats['processing_time']:.2f} seconds")
        print(f"  • Average FPS: {self.stats['avg_fps']:.2f}")

        print(f"\n🔍 Detection Statistics:")
        print(f"  • Total detections: {self.stats['total_detections']}")
        print(
            f"  • High confidence (≥{self.reid_threshold:.0%}): {self.stats['high_confidence_detections']}")
        print(
            f"  • Low confidence ({self.detection_threshold:.0%}-{self.reid_threshold:.0%}): {self.stats['low_confidence_detections']}")
        print(f"  • ReID processed: {self.stats['reid_processed']}")

        print(f"\n🧠 Clustering Results:")
        print(f"  • Clusters created: {self.stats['clusters_created']}")
        print(f"  • Persons identified: {self.stats['persons_identified']}")
        print(
            f"  • Unique persons detected: {self.stats['unique_persons_detected']}")
        print(
            f"  • Tentative assignments: {self.stats['tentative_assignments']}")
        print(
            f"  • Expected persons: {self.clustering_config['expected_persons']}")

        if self.stats['persons_identified'] > 0:
            efficiency = self.clustering_config['expected_persons'] / \
                self.stats['persons_identified']
            print(f"  • Clustering efficiency: {efficiency:.2%}")

        # Enhanced person statistics
        if self.person_galleries:
            print(f"\n🖼️  Person Analysis:")
            for person_id, crops in self.person_galleries.items():
                avg_confidence = np.mean(
                    self.person_confidences[person_id]) if self.person_confidences[person_id] else 0
                frames_appeared = len(
                    self.stats['person_detection_frames'][person_id])
                appearance_rate = frames_appeared / \
                    self.stats['total_frames'] * \
                    100 if self.stats['total_frames'] > 0 else 0

                print(f"  • {person_id}:")
                print(f"    - Crops saved: {len(crops)}")
                print(f"    - Average confidence: {avg_confidence:.2%}")
                print(
                    f"    - Frames appeared: {frames_appeared}/{self.stats['total_frames']} ({appearance_rate:.1f}%)")
                print(
                    f"    - Assigned color: {self.person_color_map.get(person_id, 'Not assigned')}")

        # Color mapping statistics
        if self.cluster_color_map:
            print(f"\n🎨 Color Mapping:")
            print(
                f"  • Cluster colors assigned: {len(self.cluster_color_map)}")
            print(f"  • Person colors assigned: {len(self.person_color_map)}")

            for cluster_id, color in self.cluster_color_map.items():
                person_id = self.cluster_to_person_map.get(
                    cluster_id, "Unknown")
                print(f"  • Cluster {cluster_id} → {person_id}: RGB{color}")

        print(f"\n⚙️  Configuration:")
        print(f"  • Detection threshold: {self.detection_threshold:.0%}")
        print(f"  • ReID threshold: {self.reid_threshold:.0%}")
        print(
            f"  • High-Confidence Similarity: ≥ {self.clustering_config['similarity_threshold']:.0%}")
        print(
            f"  • Low-Confidence Similarity: < {self.clustering_config['similarity_threshold']:.0%}")
        print(f"  • Configuration preset: three_person_dataset_90_confidence")

        print(f"\n📁 Output Files:")
        print(f"  • Annotated video: {self.stats['output_video']}")
        print(f"  • Person crops: {self.output_dir / 'crops'}")
        print(
            f"  • Statistics: {self.output_dir / 'data' / 'processing_statistics.json'}")
        print(
            f"  • Logs: {self.output_dir / 'logs' / 'video_reid_processor.log'}")

        print("\n🏆 FINAL RESULTS:")
        print(
            f"  • Total persons detected in video: {self.stats['unique_persons_detected']}")
        print(
            f"  • Clustering created {self.stats['clusters_created']} clusters")
        print(
            f"  • Expected {self.clustering_config['expected_persons']} persons")

        if self.stats['unique_persons_detected'] > 0:
            accuracy = min(
                100, (self.clustering_config['expected_persons'] / self.stats['unique_persons_detected']) * 100)
            print(f"  • Detection accuracy: {accuracy:.1f}%")

        print("="*80)

    def get_final_results(self) -> Dict[str, Any]:
        """
        Get final processing results including person count

        Returns:
            Dictionary with comprehensive results
        """
        return {
            'total_persons_detected': self.stats['unique_persons_detected'],
            'clusters_created': self.stats['clusters_created'],
            'persons_identified': self.stats['persons_identified'],
            'expected_persons': self.clustering_config['expected_persons'],
            'clustering_efficiency': (
                self.clustering_config['expected_persons'] /
                self.stats['persons_identified']
                if self.stats['persons_identified'] > 0 else 0
            ),
            'detection_accuracy': (
                min(100, (self.clustering_config['expected_persons'] /
                    self.stats['unique_persons_detected']) * 100)
                if self.stats['unique_persons_detected'] > 0 else 0
            ),
            'person_statistics': {
                person_id: {
                    'crops_saved': len(crops),
                    'average_confidence': np.mean(self.person_confidences[person_id]) if self.person_confidences[person_id] else 0,
                    'frames_appeared': len(self.stats['person_detection_frames'][person_id]),
                    'appearance_rate': len(self.stats['person_detection_frames'][person_id]) / self.stats['total_frames'] * 100 if self.stats['total_frames'] > 0 else 0,
                    'assigned_color': self.person_color_map.get(person_id)
                }
                for person_id, crops in self.person_galleries.items()
            },
            'color_mapping': {
                'cluster_colors': self.cluster_color_map,
                'person_colors': self.person_color_map
            },
            'processing_stats': self.stats
        }

    def _apply_reid_logic(self, detection: PersonDetection):
        self.stats['reid_processed'] += 1
        detection.crop_image = self.reid_system.crop_image(
            detection.bbox, frame)
        if detection.crop_image is None:
            detection.status = "nocrop"
            return

        detection.reid_features = self.reid_system.extract_features(
            detection.crop_image)
        normalized_features = detection.reid_features / \
            np.linalg.norm(detection.reid_features)

        # Check if this track_id has an established person_id
        if detection.track_id in self.track_to_person_map:
            detection.person_id = self.track_to_person_map[detection.track_id]
            detection.status = "tracked"

            # Re-check similarity to see if it's still the same person or if we should update
            search_results = self.hnsw_index.search(normalized_features, k=1)
            similarity = 1 - \
                search_results['distances'][0] if search_results and search_results['external_ids'] else 0

            if similarity < self.logic_config['reid_sim_threshold_new_id']:
                # Low similarity, lost track. Could be a new person.
                detection.status = "lost_track"
                # Forget association
                del self.track_to_person_map[detection.track_id]
                detection.person_id = None  # Reset person_id to force re-identification
                self._identify_new_detection(
                    detection, normalized_features)  # Try to re-identify
            elif similarity >= self.logic_config['embedding_update_threshold']:
                # High-confidence match, good candidate to update embedding (not implemented, just status)
                detection.status = "confirmed"
                self.stats['embedding_updates'] += 1
        else:
            # This is a new track, needs identification
            self._identify_new_detection(detection, normalized_features)

    def _identify_new_detection(self, detection: PersonDetection, features: np.ndarray):
        if self.hnsw_index.index.get_current_count() == 0:
            self._create_new_person(detection, features)
        else:
            search_results = self.hnsw_index.search(features, k=1)
            if search_results and search_results['external_ids']:
                similarity = 1 - search_results['distances'][0]
                if similarity >= self.logic_config['reid_sim_threshold_existing']:
                    # Found a match
                    detection.person_id = search_results['external_ids'][0]
                    detection.status = "re-identified"
                    self.track_to_person_map[detection.track_id] = detection.person_id
                else:
                    # Not similar enough to existing IDs, but not a new person yet. Mark as tentative.
                    detection.status = "tentative"
                    self.stats['tentative_assignments'] += 1
            else:
                # Search failed, create new person
                self._create_new_person(detection, features)

    def _create_new_person(self, detection: PersonDetection, features: np.ndarray):
        person_id = f"Person_{self.next_person_id}"
        self.hnsw_index.add_vector(features, external_id=person_id)
        detection.person_id = person_id
        detection.status = "new"
        self.track_to_person_map[detection.track_id] = person_id
        self.stats['new_persons_created'] += 1
        self.stats['persons_identified'] = self.next_person_id
        self.next_person_id += 1


def main():
    """Main function with command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Video Person Re-identification with Gaussian Clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--video",
        default="data/raw/video.mp4",
        help="Path to input video file"
    )

    parser.add_argument(
        "--output",
        default="output/test_demo_last",
        help="Output directory for results"
    )

    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        help="YOLO confidence threshold for detection display (50%%)"
    )

    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.9,
        help="YOLO confidence threshold for ReID processing (90%%)"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames to process (for testing)"
    )

    parser.add_argument(
        "--config-summary",
        action="store_true",
        help="Show configuration summary and exit"
    )

    args = parser.parse_args()

    # Show configuration summary if requested
    if args.config_summary:
        config.print_config_summary()
        sys.exit(0)

    # Validate input video
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found at {args.video}")
        print("Please provide a valid video file path.")
        sys.exit(1)

    print("="*80)
    print("VIDEO PERSON RE-ID WITH GAUSSIAN CLUSTERING")
    print("="*80)
    print(f"📹 Input video: {args.video}")
    print(f"📁 Output directory: {args.output}")
    print(f"🔍 Detection threshold: {args.detection_threshold:.0%}")
    print(f"🧠 ReID threshold: {args.reid_threshold:.0%}")
    print(f"⚙️  Configuration: three_person_dataset_90_confidence")
    if args.max_frames:
        print(f"🎬 Max frames: {args.max_frames}")
    print("="*80)

    try:
        # Initialize processor
        processor = VideoPersonReIDProcessor(
            input_video=args.video,
            output_dir=args.output,
            detection_threshold=args.detection_threshold,
            reid_threshold=args.reid_threshold,
        )

        # Process video
        print("\n🚀 Starting video processing...")
        stats = processor.process_video(max_frames=args.max_frames)

        # Print summary
        processor.print_summary()

        # Get and display final results
        final_results = processor.get_final_results()
        print(
            f"\n🎯 FINAL PERSON COUNT: {final_results['total_persons_detected']} persons detected in video")
        print(f"🔬 CLUSTERING ANALYSIS:")
        print(f"   • Clusters created: {final_results['clusters_created']}")
        print(f"   • Expected persons: {final_results['expected_persons']}")
        print(
            f"   • Clustering efficiency: {final_results['clustering_efficiency']:.2%}")
        print(
            f"   • Detection accuracy: {final_results['detection_accuracy']:.1f}%")

        return final_results

    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
