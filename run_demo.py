#!/usr/bin/env python3
"""
Advanced Person Re-identification System with Dual-Threshold Detection
======================================================================

This system implements a sophisticated two-stage detection approach:
1. YOLO Detection Stage (50% confidence): Shows bounding boxes for all person detections
2. ReID Processing Stage (90% confidence): Only processes high-confidence detections for ID assignment

Features:
- Dual-threshold detection system
- Real-time ID assignment with clustering
- Campaign-based ID confirmation
- High-quality video output with annotations
- Comprehensive logging and statistics

Usage:
    python run_demo.py --video path/to/video.mp4 --output path/to/output/
    python run_demo.py --preset three_person_dataset_90_confidence
"""

from ultralytics import YOLO
from src.clustering.hnsw_gaussian_integration import HNSWGaussianIntegration
from src.clustering.gaussian_cluster_manager import GaussianClusterManager
from src.index.hnsw_index import HNSWIndex
from src.reid.osnet_reid import OSNetReID
from config.global_config import get_config, apply_preset, config
import os
import sys
import cv2
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

# Add src to path
sys.path.append('src')

# Import configuration and pipeline modules


@dataclass
class Detection:
    """Represents a person detection with confidence and bounding box"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    person_id: Optional[str] = None
    is_confirmed: bool = False
    reid_processed: bool = False


class DualThresholdPersonReID:
    """
    Advanced Person Re-identification System with Dual-Threshold Detection

    This system separates detection and identification into two stages:
    1. Detection Stage: YOLO at 50% confidence for bounding boxes
    2. Identification Stage: ReID processing at 90% confidence for ID assignment
    """

    def __init__(self,
                 video_path: str,
                 output_dir: str = "output/dual_threshold_demo",
                 detection_confidence: float = 0.5,
                 reid_confidence: float = 0.9,
                 yolo_model: str = "yolov8s.pt",
                 reid_model: str = "osnet_x1_0",
                 config_preset: str = "three_person_dataset_90_confidence"):
        """
        Initialize the dual-threshold person re-identification system

        Args:
            video_path: Path to input video
            output_dir: Directory for output files
            detection_confidence: YOLO confidence threshold for detection (default: 0.5)
            reid_confidence: YOLO confidence threshold for ReID processing (default: 0.9)
            yolo_model: YOLO model path
            reid_model: ReID model name
            config_preset: Configuration preset to use
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.detection_confidence = detection_confidence
        self.reid_confidence = reid_confidence

        # Apply configuration preset
        if config_preset:
            apply_preset(config_preset)
            print(f"✅ Applied configuration preset: {config_preset}")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "crops").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Initialize models
        self.logger.info("🚀 Initializing detection and ReID models...")
        self.yolo_model = YOLO(yolo_model)
        self.reid_model = OSNetReID()

        # Initialize clustering system
        self.logger.info("🧠 Initializing clustering system...")
        reid_config = get_config('reid')
        clustering_config = get_config('gaussian_clustering')

        # Configure HNSW and Gaussian clustering
        hnsw_config = {
            'M': 16,
            'ef_construction': 200,
            'ef_search': 100,
            'max_elements': 1000,
            'space': 'cosine'
        }

        gaussian_config = {
            'initial_sigma_scale': 1.0,
            'mahalanobis_threshold': clustering_config['mahalanobis_threshold'],
            'min_samples_for_update': clustering_config['min_samples_per_cluster'],
            'max_clusters': clustering_config['max_clusters'],
            'fusion_threshold': 2.0,
            'split_threshold': 10.0
        }

        self.hnsw_gaussian = HNSWGaussianIntegration(
            embedding_dim=reid_config['feature_dim'],
            hnsw_config=hnsw_config,
            gaussian_config=gaussian_config,
            top_k_candidates=5
        )

        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'high_confidence_detections': 0,
            'low_confidence_detections': 0,
            'unique_persons': 0,
            'processing_times': defaultdict(list),
            'detection_history': [],
            'reid_assignments': {}
        }

        # ID management
        self.next_person_id = 1
        self.person_galleries = defaultdict(list)
        self.person_confidences = defaultdict(list)

        self.logger.info("✅ System initialized successfully!")

    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_file = self.output_dir / "logs" / "dual_threshold_pipeline.log"

        # Create logger
        self.logger = logging.getLogger("DualThresholdReID")
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

    def detect_persons(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in frame using YOLO with detection confidence threshold

        Args:
            frame: Input frame

        Returns:
            List of Detection objects
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
                    if confidence >= self.detection_confidence:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        detection = Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            reid_processed=confidence >= self.reid_confidence
                        )
                        detections.append(detection)

        # Record timing
        detection_time = time.time() - start_time
        self.stats['processing_times']['detection'].append(detection_time)

        return detections

    def process_high_confidence_detections(self, frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
        """
        Process high-confidence detections through ReID pipeline

        Args:
            frame: Input frame
            detections: List of detections

        Returns:
            Updated detections with person IDs
        """
        start_time = time.time()

        high_conf_detections = [
            d for d in detections if d.confidence >= self.reid_confidence]

        if not high_conf_detections:
            return detections

        # Extract crops for high-confidence detections
        crops = []
        valid_detections = []

        for detection in high_conf_detections:
            x1, y1, x2, y2 = detection.bbox

            # Validate crop dimensions
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
                    valid_detections.append(detection)

        if not crops:
            return detections

        # Extract ReID features
        features = []
        for crop in crops:
            try:
                feature = self.reid_model.extract_features(crop)[0]
                features.append(feature)
            except Exception as e:
                self.logger.warning(f"Failed to extract features: {e}")
                features.append(None)

        # Assign person IDs using clustering
        for i, (detection, feature) in enumerate(zip(valid_detections, features)):
            if feature is not None:
                try:
                    # Use HNSW-Gaussian integration for ID assignment
                    cluster_id, mahalanobis_distance, assignment_info = self.hnsw_gaussian.assign_identity(
                        feature)

                    # Convert cluster ID to person ID
                    if cluster_id not in self.stats['reid_assignments']:
                        self.stats['reid_assignments'][cluster_id] = f"Person_{self.next_person_id}"
                        self.next_person_id += 1
                        self.stats['unique_persons'] += 1

                    detection.person_id = self.stats['reid_assignments'][cluster_id]
                    detection.is_confirmed = True
                    detection.reid_processed = True

                    # Save crop to gallery
                    person_id = detection.person_id
                    self.person_galleries[person_id].append(crops[i])
                    self.person_confidences[person_id].append(
                        detection.confidence)

                    # Save crop to disk
                    self.save_person_crop(
                        crops[i], person_id, self.stats['total_frames'])

                except Exception as e:
                    self.logger.warning(f"Failed to assign ID: {e}")
                    detection.person_id = "Unknown"

        # Record timing
        reid_time = time.time() - start_time
        self.stats['processing_times']['reid'].append(reid_time)

        return detections

    def save_person_crop(self, crop: np.ndarray, person_id: str, frame_num: int):
        """Save person crop to disk"""
        person_dir = self.output_dir / "crops" / person_id
        person_dir.mkdir(exist_ok=True)

        crop_path = person_dir / f"frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(crop_path), crop)

    def annotate_frame(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Annotate frame with bounding boxes and person IDs

        Args:
            frame: Input frame
            detections: List of detections

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            confidence = detection.confidence

            # Determine color and label based on confidence and ID status
            if detection.reid_processed and detection.person_id:
                # High confidence with ID
                color = (0, 255, 0)  # Green
                label = f"{detection.person_id} ({confidence:.2f})"
            elif detection.confidence >= self.reid_confidence:
                # High confidence but no ID yet
                color = (0, 255, 255)  # Yellow
                label = f"Processing... ({confidence:.2f})"
            else:
                # Low confidence - show question mark
                color = (0, 0, 255)  # Red
                label = f"? ({confidence:.2f})"

            # Draw bounding box
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

        # Add system info
        info_text = [
            f"Frame: {self.stats['total_frames']}",
            f"Detections: {len(detections)}",
            f"High Conf: {sum(1 for d in detections if d.confidence >= self.reid_confidence)}",
            f"Unique Persons: {self.stats['unique_persons']}",
            f"Detection Threshold: {self.detection_confidence:.0%}",
            f"ReID Threshold: {self.reid_confidence:.0%}"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(annotated_frame, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_frame

    def process_video(self, max_frames: Optional[int] = None) -> Dict:
        """
        Process video with dual-threshold detection system

        Args:
            max_frames: Maximum frames to process (None for all)

        Returns:
            Processing statistics
        """
        self.logger.info(f"🎬 Starting video processing: {self.video_path}")

        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

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
        output_video_path = self.output_dir / "videos" / \
            f"dual_threshold_output_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path),
                              fourcc, fps, (width, height))

        # Processing loop
        start_time = time.time()
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_count >= max_frames):
                    break

                frame_start_time = time.time()

                # Stage 1: Detect persons (50% confidence)
                detections = self.detect_persons(frame)
                self.stats['total_detections'] += len(detections)

                # Count detection types
                high_conf_count = sum(
                    1 for d in detections if d.confidence >= self.reid_confidence)
                low_conf_count = len(detections) - high_conf_count

                self.stats['high_confidence_detections'] += high_conf_count
                self.stats['low_confidence_detections'] += low_conf_count

                # Stage 2: Process high-confidence detections (90% confidence)
                detections = self.process_high_confidence_detections(
                    frame, detections)

                # Stage 3: Annotate frame
                annotated_frame = self.annotate_frame(frame, detections)

                # Write frame
                out.write(annotated_frame)

                # Update statistics
                frame_time = time.time() - frame_start_time
                self.stats['processing_times']['frame'].append(frame_time)
                self.stats['total_frames'] = frame_count + 1

                # Log progress
                if frame_count % 30 == 0:
                    current_fps = 1.0 / frame_time if frame_time > 0 else 0
                    self.logger.info(f"Frame {frame_count}/{total_frames} | "
                                     f"Detections: {len(detections)} | "
                                     f"High Conf: {high_conf_count} | "
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

        self.stats.update({
            'processing_time': processing_time,
            'avg_fps': avg_fps,
            'output_video': str(output_video_path)
        })

        # Save statistics
        self.save_statistics()

        self.logger.info(
            f"✅ Processing completed: {frame_count} frames in {processing_time:.2f}s")
        self.logger.info(f"📊 Average FPS: {avg_fps:.2f}")
        self.logger.info(f"🎥 Output video: {output_video_path}")

        return self.stats

    def save_statistics(self):
        """Save processing statistics to JSON file"""
        stats_path = self.output_dir / "data" / "processing_stats.json"

        # Prepare statistics for JSON serialization
        json_stats = {
            'total_frames': self.stats['total_frames'],
            'total_detections': self.stats['total_detections'],
            'high_confidence_detections': self.stats['high_confidence_detections'],
            'low_confidence_detections': self.stats['low_confidence_detections'],
            'unique_persons': self.stats['unique_persons'],
            'processing_time': self.stats['processing_time'],
            'avg_fps': self.stats['avg_fps'],
            'detection_confidence': self.detection_confidence,
            'reid_confidence': self.reid_confidence,
            'output_video': self.stats['output_video'],
            'person_galleries': {k: len(v) for k, v in self.person_galleries.items()},
            'processing_times': {
                stage: {
                    'mean': np.mean(times) if times else 0,
                    'std': np.std(times) if times else 0,
                    'min': np.min(times) if times else 0,
                    'max': np.max(times) if times else 0
                }
                for stage, times in self.stats['processing_times'].items()
            }
        }

        with open(stats_path, 'w') as f:
            json.dump(json_stats, f, indent=2)

        self.logger.info(f"📊 Statistics saved to: {stats_path}")

    def get_person_gallery(self, person_id: str, max_crops: int = 10) -> List[np.ndarray]:
        """Get person gallery crops"""
        crops = self.person_galleries.get(person_id, [])
        return crops[:max_crops]

    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*80)
        print("DUAL-THRESHOLD PERSON RE-ID SYSTEM - PROCESSING SUMMARY")
        print("="*80)

        print(f"📹 Video: {self.video_path}")
        print(f"📊 Total frames processed: {self.stats['total_frames']}")
        print(
            f"⏱️  Processing time: {self.stats['processing_time']:.2f} seconds")
        print(f"🚀 Average FPS: {self.stats['avg_fps']:.2f}")

        print(f"\n🔍 Detection Statistics:")
        print(f"  • Total detections: {self.stats['total_detections']}")
        print(
            f"  • High confidence (≥{self.reid_confidence:.0%}): {self.stats['high_confidence_detections']}")
        print(
            f"  • Low confidence ({self.detection_confidence:.0%}-{self.reid_confidence:.0%}): {self.stats['low_confidence_detections']}")

        print(f"\n👥 Person Identification:")
        print(f"  • Unique persons identified: {self.stats['unique_persons']}")
        print(f"  • Detection threshold: {self.detection_confidence:.0%}")
        print(f"  • ReID processing threshold: {self.reid_confidence:.0%}")

        if self.person_galleries:
            print(f"\n🖼️  Person Galleries:")
            for person_id, crops in self.person_galleries.items():
                avg_confidence = np.mean(self.person_confidences[person_id])
                print(
                    f"  • {person_id}: {len(crops)} crops (avg confidence: {avg_confidence:.2%})")

        print(f"\n📁 Output Files:")
        print(f"  • Video: {self.stats['output_video']}")
        print(f"  • Crops: {self.output_dir / 'crops'}")
        print(
            f"  • Statistics: {self.output_dir / 'data' / 'processing_stats.json'}")
        print(
            f"  • Logs: {self.output_dir / 'logs' / 'dual_threshold_pipeline.log'}")

        print("="*80)


def main():
    """Main function with command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dual-Threshold Person Re-identification System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--video",
        default="data/raw/video.mp4",
        help="Path to input video file"
    )

    parser.add_argument(
        "--output",
        default="output/dual_threshold_demo",
        help="Output directory for results"
    )

    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.5,
        help="YOLO confidence threshold for detection (bounding boxes)"
    )

    parser.add_argument(
        "--reid-confidence",
        type=float,
        default=0.9,
        help="YOLO confidence threshold for ReID processing"
    )

    parser.add_argument(
        "--preset",
        choices=["high_accuracy", "high_speed", "balanced",
                 "three_person_dataset", "three_person_dataset_90_confidence"],
        default="three_person_dataset_90_confidence",
        help="Configuration preset to use"
    )

    parser.add_argument(
        "--yolo-model",
        default="yolov8s.pt",
        help="YOLO model path"
    )

    parser.add_argument(
        "--reid-model",
        default="osnet_x1_0",
        help="ReID model name"
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
    print("DUAL-THRESHOLD PERSON RE-IDENTIFICATION SYSTEM")
    print("="*80)
    print(f"📹 Input video: {args.video}")
    print(f"📁 Output directory: {args.output}")
    print(f"🔍 Detection confidence: {args.detection_confidence:.0%}")
    print(f"🧠 ReID confidence: {args.reid_confidence:.0%}")
    print(f"🎯 YOLO model: {args.yolo_model}")
    print(f"🤖 ReID model: {args.reid_model}")
    print(f"⚙️  Configuration preset: {args.preset}")
    if args.max_frames:
        print(f"🎬 Max frames: {args.max_frames}")
    print("="*80)

    try:
        # Initialize system
        system = DualThresholdPersonReID(
            video_path=args.video,
            output_dir=args.output,
            detection_confidence=args.detection_confidence,
            reid_confidence=args.reid_confidence,
            yolo_model=args.yolo_model,
            reid_model=args.reid_model,
            config_preset=args.preset
        )

        # Process video
        print("\n🚀 Starting video processing...")
        stats = system.process_video(max_frames=args.max_frames)

        # Print summary
        system.print_summary()

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
