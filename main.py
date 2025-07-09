#!/usr/bin/env python3
"""
Main script for Person Re-identification System.

This script orchestrates the complete pipeline:
1. Person detection using YOLO11m-pose
2. Embedding extraction for robust representation
3. Re-identification using deep learning models
4. Persistent tracking across video frames
5. Visualization and output generation
"""

import argparse
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from tqdm import tqdm

# Import our modules
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.utils.video_utils import VideoProcessor
from src.detection.person_detector import PersonDetector
from src.embedding.embedding_extractor import EmbeddingExtractor
from src.reid.reid_matcher import ReIDMatcher
from src.tracking.track_manager import TrackManager
from src.visualization.visualizer import Visualizer


class PersonReIDSystem:
    """Main system for person re-identification."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the person re-identification system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = ConfigLoader(config_path)
        self.config.validate_config()
        
        # Setup logging
        log_level = self.config.get("logging.level", "INFO")
        log_file = self.config.get("logging.log_file", "logs/reid_system.log")
        setup_logger(log_level, log_file)
        
        logging.info("Initializing Person Re-identification System")
        
        # Initialize components
        self._initialize_components()
        
        logging.info("System initialization complete")
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Person detector
        self.detector = PersonDetector(self.config)
        
        # Embedding extractor
        self.embedding_extractor = EmbeddingExtractor(self.config)
        
        # Re-identification matcher
        self.reid_matcher = ReIDMatcher(self.config, self.embedding_extractor)
        
        # Track manager
        self.track_manager = TrackManager(self.config, self.reid_matcher)
        
        # Visualizer
        self.visualizer = Visualizer(self.config)
        
        # Video processor
        self.video_processor = VideoProcessor(self.config)
    
    def process_video(self, input_path: str, output_path: str) -> dict:
        """
        Process a video file for person re-identification.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            
        Returns:
            Dictionary with processing statistics
        """
        logging.info(f"Processing video: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logging.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        video_config = self.config.get_video_config()
        fourcc = cv2.VideoWriter_fourcc(*video_config.get("output_codec", "mp4v"))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'processing_time': 0.0,
            'fps_processing': 0.0
        }
        
        # Process frames
        start_time = time.time()
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            frame_id = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                timestamp = frame_id / fps
                tracking_result = self._process_frame(frame, frame_id, timestamp)
                
                # Visualize results
                annotated_frame = self.visualizer.draw_frame(frame, tracking_result)
                
                # Write frame
                out.write(annotated_frame)
                
                # Update statistics
                stats['processed_frames'] += 1
                stats['total_detections'] += len(tracking_result.persons)
                stats['total_tracks'] = len(tracking_result.tracks)
                
                frame_id += 1
                pbar.update(1)
                
                # Update progress bar description
                pbar.set_description(f"Frame {frame_id}/{total_frames} | Tracks: {stats['total_tracks']}")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Calculate final statistics
        end_time = time.time()
        stats['processing_time'] = end_time - start_time
        stats['fps_processing'] = stats['processed_frames'] / stats['processing_time']
        
        # Get track statistics
        track_stats = self.track_manager.get_track_statistics()
        stats.update(track_stats)
        
        logging.info(f"Video processing complete. Output: {output_path}")
        logging.info(f"Processing statistics: {stats}")
        
        return stats
    
    def _process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float):
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            frame_id: Frame identifier
            timestamp: Frame timestamp
            
        Returns:
            Tracking result
        """
        # 1. Detect persons
        detections = self.detector.detect(frame, frame_id, timestamp)
        
        # 2. Extract person crops and embeddings
        persons = []
        for detection in detections:
            # Extract person crop
            person_crop = self._extract_person_crop(frame, detection)
            
            # Extract embedding
            embedding = self.embedding_extractor.extract_embedding(
                person_crop, detection.pose_keypoints
            )
            
            # Create person object
            person = Person(
                detection=detection,
                embedding=embedding
            )
            persons.append(person)
        
        # 3. Update tracks
        tracking_result = self.track_manager.update_tracks(frame_id, timestamp, persons)
        
        return tracking_result
    
    def _extract_person_crop(self, frame: np.ndarray, detection) -> np.ndarray:
        """
        Extract person crop from frame.
        
        Args:
            frame: Input frame
            detection: Person detection
            
        Returns:
            Person crop
        """
        x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
        
        # Ensure coordinates are within frame bounds
        height, width = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        
        # Handle empty crop
        if crop.size == 0:
            # Return a small black image as fallback
            return np.zeros((64, 32, 3), dtype=np.uint8)
        
        return crop
    
    def save_results(self, output_dir: str, stats: dict):
        """
        Save processing results and statistics.
        
        Args:
            output_dir: Output directory
            stats: Processing statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save statistics
        import json
        stats_file = output_path / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save track information
        tracks_file = output_path / "tracks_info.json"
        tracks_data = []
        
        for track in self.track_manager.tracks.values():
            track_info = {
                'track_id': track.track_id,
                'person_id': track.person_id,
                'total_frames': track.total_frames,
                'duration': track.get_duration(),
                'first_seen': track.first_seen,
                'last_seen': track.last_seen,
                'is_active': track.is_active
            }
            tracks_data.append(track_info)
        
        with open(tracks_file, 'w') as f:
            json.dump(tracks_data, f, indent=2)
        
        logging.info(f"Results saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Person Re-identification System")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output video path")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--output-dir", "-d", default="data/processed", help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = PersonReIDSystem(args.config)
        
        # Process video
        stats = system.process_video(args.input, args.output)
        
        # Save results
        system.save_results(args.output_dir, stats)
        
        print(f"\nProcessing complete!")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        print(f"Processing FPS: {stats['fps_processing']:.2f}")
        print(f"Total tracks: {stats['total_tracks']}")
        print(f"Active tracks: {stats['active_tracks']}")
        
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main() 