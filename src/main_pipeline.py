#!/usr/bin/env python3
"""
Main Pipeline for Person Re-identification System
Integrates YOLO detection, Huffman tracking, OSNet ReID, HNSW indexing, and ID fusion
"""

from config.global_config import get_config, apply_preset
import cv2
import numpy as np
import os
import sys
import json
from typing import Dict, List, Optional, Tuple
import time
import logging
from pathlib import Path

# Import custom modules
from src.detection.infer import YOLODetector
from src.tracking.huffman_tracker import PersonTracker
from src.reid.osnet_reid import OSNetReID
from src.index.hnsw_index import HNSWIndex
from src.fusion.fusion import IDFusion


class PersonReIDPipeline:
    """Complete person re-identification pipeline"""

    def __init__(self,
                 yolo_model_path: str = None,
                 reid_model_name: str = None,
                 hnsw_dim: int = None,
                 hnsw_max_elements: int = None,
                 output_dir: str = None,
                 save_crops: bool = None,
                 save_video: bool = None,
                 device: str = None,
                 config_preset: str = None):
        """
        Initialize the complete pipeline

        Args:
            yolo_model_path: Path to YOLO model (uses config default if None)
            reid_model_name: Name of ReID model (uses config default if None)
            hnsw_dim: Dimensionality of HNSW index (uses config default if None)
            hnsw_max_elements: Maximum elements in HNSW index (uses config default if None)
            output_dir: Output directory for results (uses config default if None)
            save_crops: Whether to save person crops (uses config default if None)
            save_video: Whether to save annotated video (uses config default if None)
            device: Device to use (uses config default if None)
            config_preset: Configuration preset to apply ('high_accuracy', 'high_speed', 'balanced')
        """
        # Apply configuration preset if specified
        if config_preset:
            apply_preset(config_preset)

        # Get configurations
        self.detection_config = get_config('detection')
        self.tracking_config = get_config('tracking')
        self.reid_config = get_config('reid')
        self.indexing_config = get_config('indexing')
        self.fusion_config = get_config('fusion')
        self.pipeline_config = get_config('pipeline')
        self.output_config = get_config('output')
        self.performance_config = get_config('performance')
        self.paths_config = get_config('paths')

        # Use provided values or defaults from config
        self.yolo_model_path = yolo_model_path or self.detection_config['model_path']
        self.reid_model_name = reid_model_name or self.reid_config['model_name']
        self.hnsw_dim = hnsw_dim or self.indexing_config['dimension']
        self.hnsw_max_elements = hnsw_max_elements or self.indexing_config['max_elements']
        self.device = device or self.reid_config['device']
        self.save_crops = save_crops if save_crops is not None else self.output_config[
            'save_crops']
        self.save_video = save_video if save_video is not None else self.output_config[
            'save_video']

        # Setup output directory
        output_dir = output_dir or str(self.paths_config['output_dir'])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "crops").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)

        self.save_crops = save_crops
        self.save_video = save_video

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "pipeline.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize modules
        self.logger.info("Initializing pipeline modules...")

        try:
            # Detection module - uses config defaults
            self.detector = YOLODetector(
                model_path=self.yolo_model_path
            )
            self.logger.info("✓ YOLO detector initialized")

            # Tracking module - uses config defaults
            self.tracker = PersonTracker()
            self.logger.info("✓ Huffman tracker initialized")

            # ReID module - uses config defaults
            self.reid = OSNetReID(
                model_name=self.reid_model_name,
                device=self.device
            )
            self.logger.info("✓ OSNet ReID initialized")

            # HNSW index - uses config defaults
            self.index = HNSWIndex(
                dim=self.hnsw_dim,
                max_elements=self.hnsw_max_elements
            )
            self.logger.info("✓ HNSW index initialized")

            # Fusion module - uses config defaults
            self.fusion = IDFusion()
            self.logger.info("✓ ID fusion initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise

        # Pipeline state
        self.frame_count = 0
        self.total_detections = 0
        self.total_tracks = 0
        self.processing_times = {
            'detection': [],
            'tracking': [],
            'reid': [],
            'indexing': [],
            'fusion': [],
            'total': []
        }

        # Color palette for visualization
        self.colors = self._generate_colors(50)

    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization"""
        np.random.seed(42)  # For reproducible colors
        colors = []
        for _ in range(num_colors):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors

    def _save_person_crop(self, image: np.ndarray, bbox: List[int],
                          global_id: int, frame_idx: int):
        """Save person crop to disk"""
        if not self.save_crops:
            return

        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]

        if crop.size > 0:
            crop_dir = self.output_dir / "crops" / f"person_{global_id:03d}"
            crop_dir.mkdir(parents=True, exist_ok=True)

            crop_path = crop_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(crop_path), crop)

    def _annotate_frame(self, frame: np.ndarray,
                        detections: List[Dict],
                        tracks: Dict[int, Dict],
                        assignments: Dict[int, int]) -> np.ndarray:
        """Annotate frame with detections, tracks, and global IDs"""
        annotated = frame.copy()

        # Draw detections and tracks
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            # Find corresponding track
            track_id = None
            global_id = None

            for tid, track_info in tracks.items():
                track_bbox = track_info['bbox']
                # Simple overlap check
                if (abs(bbox[0] - track_bbox[0]) < 20 and
                        abs(bbox[1] - track_bbox[1]) < 20):
                    track_id = tid
                    global_id = assignments.get(tid)
                    break

            # Choose color based on global ID
            if global_id is not None:
                color = self.colors[global_id % len(self.colors)]
            else:
                color = (128, 128, 128)  # Gray for unassigned

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            label_parts = []
            if global_id is not None:
                label_parts.append(f"ID:{global_id}")
            if track_id is not None:
                label_parts.append(f"T:{track_id}")

            conf = detection.get('confidence', 0.0)
            label_parts.append(f"{conf:.2f}")

            label = " ".join(label_parts)

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10),
                          (x1 + label_w, y1), color, -1)

            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame through the complete pipeline

        Args:
            frame: Input frame

        Returns:
            Tuple of (annotated_frame, frame_info)
        """
        frame_start_time = time.time()
        frame_info = {
            'frame_idx': self.frame_count,
            'detections': [],
            'tracks': {},
            'reid_features': {},
            'hnsw_results': {},
            'assignments': {},
            'processing_times': {}
        }

        # 1. Detection
        detection_start = time.time()
        detection_results = self.detector.detect_persons(frame)
        detections = detection_results['detections']
        detection_time = time.time() - detection_start
        self.processing_times['detection'].append(detection_time)
        frame_info['processing_times']['detection'] = detection_time

        self.total_detections += len(detections)
        frame_info['detections'] = detections

        if not detections:
            # No detections, just update tracker
            self.tracker.update([])
            annotated = frame.copy()
            total_time = time.time() - frame_start_time
            self.processing_times['total'].append(total_time)
            frame_info['processing_times']['total'] = total_time
            return annotated, frame_info

        # 2. Tracking
        tracking_start = time.time()

        # Convert detections to format expected by tracker
        detection_boxes = []
        for det in detections:
            bbox = det['bbox']
            detection_boxes.append({
                'bbox': bbox,
                'confidence': det['confidence']
            })

        tracks = self.tracker.update(detection_boxes)
        tracking_time = time.time() - tracking_start
        self.processing_times['tracking'].append(tracking_time)
        frame_info['processing_times']['tracking'] = tracking_time

        self.total_tracks = len(tracks)
        frame_info['tracks'] = tracks

        if not tracks:
            annotated = self._annotate_frame(frame, detections, {}, {})
            total_time = time.time() - frame_start_time
            self.processing_times['total'].append(total_time)
            frame_info['processing_times']['total'] = total_time
            return annotated, frame_info

        # 3. ReID Feature Extraction
        reid_start = time.time()
        reid_features = {}

        # Extract crops for each track
        crops = []
        track_ids = []

        for track_id, track_info in tracks.items():
            bbox = track_info['bbox']
            x1, y1, x2, y2 = bbox

            # Ensure bbox is within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                track_ids.append(track_id)

        if crops:
            # Batch feature extraction
            features = self.reid.extract_features_batch(crops)
            for i, track_id in enumerate(track_ids):
                reid_features[track_id] = features[i]

        reid_time = time.time() - reid_start
        self.processing_times['reid'].append(reid_time)
        frame_info['processing_times']['reid'] = reid_time
        frame_info['reid_features'] = {k: v.tolist()
                                       for k, v in reid_features.items()}

        # 4. HNSW Index Search
        indexing_start = time.time()
        hnsw_results = {}

        for track_id, features in reid_features.items():
            # Search for similar features
            search_result = self.index.search(features, k=5)
            hnsw_results[track_id] = search_result

        indexing_time = time.time() - indexing_start
        self.processing_times['indexing'].append(indexing_time)
        frame_info['processing_times']['indexing'] = indexing_time
        frame_info['hnsw_results'] = hnsw_results

        # 5. ID Fusion
        fusion_start = time.time()
        assignments = self.fusion.fuse_identities(
            tracks, reid_features, hnsw_results, self.frame_count
        )
        fusion_time = time.time() - fusion_start
        self.processing_times['fusion'].append(fusion_time)
        frame_info['processing_times']['fusion'] = fusion_time
        frame_info['assignments'] = assignments

        # 6. Update HNSW index with new features
        for track_id, features in reid_features.items():
            if track_id in assignments:
                global_id = assignments[track_id]
                external_id = f"person_{global_id}"
                self.index.add_vector(features, external_id)

        # 7. Save person crops
        if self.save_crops:
            for track_id, track_info in tracks.items():
                if track_id in assignments:
                    global_id = assignments[track_id]
                    self._save_person_crop(frame, track_info['bbox'],
                                           global_id, self.frame_count)

        # 8. Annotate frame
        annotated = self._annotate_frame(
            frame, detections, tracks, assignments)

        # Add frame info
        info_text = [
            f"Frame: {self.frame_count}",
            f"Detections: {len(detections)}",
            f"Tracks: {len(tracks)}",
            f"Active IDs: {len(set(assignments.values()))}",
            f"FPS: {1.0/detection_time:.1f}"
        ]

        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(annotated, text, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total_time = time.time() - frame_start_time
        self.processing_times['total'].append(total_time)
        frame_info['processing_times']['total'] = total_time

        return annotated, frame_info

    def process_video(self, video_path: str,
                      output_path: Optional[str] = None,
                      max_frames: Optional[int] = None) -> Dict:
        """
        Process complete video through pipeline

        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
            max_frames: Maximum frames to process (optional)

        Returns:
            Processing statistics
        """
        self.logger.info(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(
            f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Setup output video writer
        if output_path is None:
            output_path = self.output_dir / "videos" / \
                f"output_{Path(video_path).stem}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if self.save_video:
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height))

        # Process frames
        frame_infos = []
        self.frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and self.frame_count >= max_frames:
                    break

                # Process frame
                annotated, frame_info = self.process_frame(frame)
                frame_infos.append(frame_info)

                # Write output frame
                if out is not None:
                    out.write(annotated)

                # Progress logging
                if self.frame_count % 100 == 0:
                    self.logger.info(
                        f"Processed {self.frame_count}/{total_frames} frames")

                self.frame_count += 1

        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")

        finally:
            cap.release()
            if out is not None:
                out.release()

        # Generate statistics
        stats = self._generate_statistics(frame_infos)

        # Save results
        self._save_results(frame_infos, stats)

        self.logger.info("Video processing completed")
        return stats

    def _generate_statistics(self, frame_infos: List[Dict]) -> Dict:
        """Generate processing statistics"""
        stats = {
            'total_frames': len(frame_infos),
            'total_detections': self.total_detections,
            'avg_detections_per_frame': self.total_detections / max(len(frame_infos), 1),
            'max_tracks': self.total_tracks,
            'fusion_stats': self.fusion.get_statistics(),
            'processing_times': {}
        }

        # Compute timing statistics
        for stage, times in self.processing_times.items():
            if times:
                stats['processing_times'][stage] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times)
                }

        # Compute FPS
        if stats['processing_times'].get('total', {}).get('mean'):
            stats['avg_fps'] = 1.0 / stats['processing_times']['total']['mean']
        else:
            stats['avg_fps'] = 0.0

        return stats

    def _save_results(self, frame_infos: List[Dict], stats: Dict):
        """Save processing results to disk"""

        # Save frame-by-frame results
        results_path = self.output_dir / "data" / "frame_results.json"
        with open(results_path, 'w') as f:
            json.dump(frame_infos, f, indent=2)

        # Save statistics
        stats_path = self.output_dir / "data" / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Save fusion state
        fusion_path = self.output_dir / "data" / "fusion_state.json"
        self.fusion.save_state(str(fusion_path))

        # Save HNSW index
        index_path = self.output_dir / "data" / "hnsw_index.bin"
        self.index.save_index(str(index_path))

        self.logger.info(f"Results saved to {self.output_dir}")

    def get_person_gallery(self, global_id: int, max_crops: int = 10) -> List[str]:
        """
        Get gallery of crops for a specific person

        Args:
            global_id: Global person ID
            max_crops: Maximum number of crops to return

        Returns:
            List of crop file paths
        """
        crop_dir = self.output_dir / "crops" / f"person_{global_id:03d}"
        if not crop_dir.exists():
            return []

        crop_files = sorted(crop_dir.glob("*.jpg"))
        return [str(f) for f in crop_files[:max_crops]]


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Person Re-identification Pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--yolo-model", default="yolov8n.pt",
                        help="YOLO model path")
    parser.add_argument("--reid-model", default="osnet_x1_0",
                        help="ReID model name")
    parser.add_argument("--max-frames", type=int,
                        help="Maximum frames to process")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--no-crops", action="store_true",
                        help="Don't save crops")
    parser.add_argument("--no-video", action="store_true",
                        help="Don't save video")

    args = parser.parse_args()

    # Create pipeline
    pipeline = PersonReIDPipeline(
        yolo_model_path=args.yolo_model,
        reid_model_name=args.reid_model,
        output_dir=args.output,
        save_crops=not args.no_crops,
        save_video=not args.no_video,
        device=args.device
    )

    # Process video
    stats = pipeline.process_video(
        video_path=args.video,
        max_frames=args.max_frames
    )

    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total frames: {stats['total_frames']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average FPS: {stats['avg_fps']:.2f}")
    print(f"Active persons: {stats['fusion_stats']['active_persons']}")
    print(f"Total identities: {stats['fusion_stats']['next_global_id']}")
    print("="*50)


if __name__ == "__main__":
    main()
