#!/usr/bin/env python3
"""
Test script for persistent person tracking.
"""

import cv2
import numpy as np
import time
from pathlib import Path

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.detection.person_detector import PersonDetector
from src.embedding.embedding_extractor import EmbeddingExtractor
from src.reid.reid_matcher import ReIDMatcher
from src.tracking.persistent_tracker import PersistentTracker
from src.visualization.visualizer import Visualizer


def test_persistent_tracking():
    """Test persistent tracking functionality."""
    
    # Setup logging
    setup_logger("INFO", "logs/test_persistent.log")
    
    # Load configuration
    config = ConfigLoader("config/config.yaml")
    
    # Initialize components
    detector = PersonDetector(config)
    embedding_extractor = EmbeddingExtractor(config)
    reid_matcher = ReIDMatcher(config, embedding_extractor)
    tracker = PersistentTracker(config, reid_matcher)
    visualizer = Visualizer(config)
    
    # Test video path
    video_path = "data/raw/video_1.mp4"
    
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Testing persistent tracking on {total_frames} frames at {fps} FPS")
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        timestamp = frame_count / fps
        
        # Detect persons
        detections = detector.detect(frame, frame_count, timestamp)
        
        # Extract embeddings
        persons = []
        for detection in detections:
            # Extract person crop (simplified)
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            
            if x2 > x1 and y2 > y1:
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    embedding = embedding_extractor.extract_embedding(person_crop, detection.pose_keypoints)
                    
                    from src.utils.data_structures import Person
                    person = Person(
                        detection=detection,
                        embedding=embedding
                    )
                    persons.append(person)
        
        # Update tracks
        tracking_result = tracker.update_tracks(frame_count, timestamp, persons)
        
        # Print statistics every 100 frames
        if frame_count % 100 == 0:
            stats = tracker.get_track_statistics()
            print(f"Frame {frame_count}: Active tracks: {stats['active_tracks']}, "
                  f"Total persons: {stats['total_persons']}, "
                  f"ReID matches: {stats['total_reid_matches']}")
        
        frame_count += 1
        
        # Limit processing for testing
        if frame_count >= 300:  # Process first 300 frames
            break
    
    # End session
    tracker.end_session(time.time())
    
    # Print final statistics
    stats = tracker.get_track_statistics()
    print("\n=== Final Statistics ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Active tracks: {stats['active_tracks']}")
    print(f"Total unique persons: {stats['total_persons']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"ReID matches: {stats['total_reid_matches']}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    
    # Export history
    tracker.export_history_json("data/processed/test_history.json")
    print("History exported to data/processed/test_history.json")
    
    # Cleanup
    cap.release()


if __name__ == "__main__":
    test_persistent_tracking() 