#!/usr/bin/env python3
"""
Clean tracking test with visualization - starts fresh without history.
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.detection.person_detector import PersonDetector
from src.embedding.embedding_extractor import EmbeddingExtractor
from src.reid.reid_matcher import ReIDMatcher
from src.tracking.persistent_tracker import PersistentTracker
from src.visualization.visualizer import Visualizer


def test_clean_tracking_visual():
    """Test tracking with clean history and visualization."""
    
    # Clean up existing history
    history_file = Path("data/persistent/person_history.pkl")
    if history_file.exists():
        os.remove(history_file)
        print("🧹 Cleaned existing history")
    
    # Setup logging
    setup_logger("INFO", "logs/clean_tracking_visual.log")
    
    # Load configuration
    config = ConfigLoader("config/config.yaml")
    
    # Initialize components
    detector = PersonDetector(config)
    embedding_extractor = EmbeddingExtractor(config)
    reid_matcher = ReIDMatcher(config, embedding_extractor)
    tracker = PersistentTracker(config, reid_matcher)
    visualizer = Visualizer(config)
    
    # Test video path
    video_path = "data/raw/video.mp4"
    output_path = "data/processed/output_tracking_clean.mp4"
    
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Clean tracking test with visualization on {total_frames} frames at {fps} FPS")
    print(f"Video size: {width}x{height}")
    print("=" * 50)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Track ID consistency
    id_changes = 0
    current_ids = {}  # track_id -> person_id mapping
    person_count = 0
    
    # Process frames
    frame_count = 0
    
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
        
        # Check for ID changes
        for track in tracking_result.tracks:
            if track.track_id in current_ids:
                old_person_id = current_ids[track.track_id]
                if old_person_id != track.person_id:
                    id_changes += 1
                    print(f"⚠️  ID CHANGE at frame {frame_count}: Track {track.track_id} "
                          f"changed from {old_person_id} to {track.person_id}")
            else:
                # New track
                person_count = max(person_count, track.person_id + 1)
            current_ids[track.track_id] = track.person_id
        
        # Visualize frame
        visualized_frame = visualizer.draw_frame(frame, tracking_result)
        
        # Write frame to video
        out.write(visualized_frame)
        
        # Show frame (optional - comment out if you don't want to see it)
        cv2.imshow('Clean Tracking Test', visualized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Print info every 25 frames
        if frame_count % 25 == 0:
            stats = tracker.get_track_statistics()
            locked_info = tracker.get_locked_ids_info()
            print(f"Frame {frame_count}: Active tracks: {stats['active_tracks']}, "
                  f"Total persons: {stats['total_persons']}, "
                  f"Locked IDs: {len(locked_info['locked_persons'])}")
            
            # Print current tracks
            for track in tracking_result.tracks:
                if track.is_active:
                    is_locked = track.track_id in locked_info['locked_persons']
                    lock_status = "🔒" if is_locked else "🔓"
                    print(f"    Track {track.track_id} -> Person {track.person_id} {lock_status}")
        
        frame_count += 1
        
        # Limit processing for testing
        if frame_count >= 200:  # Process first 200 frames
            break
    
    # End session
    tracker.end_session(time.time())
    
    # Print final statistics
    stats = tracker.get_track_statistics()
    locked_info = tracker.get_locked_ids_info()
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Frames processed: {frame_count}")
    print(f"Active tracks: {stats['active_tracks']}")
    print(f"Total unique persons: {stats['total_persons']}")
    print(f"Person count: {person_count}")
    print(f"ID changes detected: {id_changes}")
    print(f"Locked IDs: {len(locked_info['locked_persons'])}")
    print(f"Output video saved to: {output_path}")
    
    if locked_info['locked_persons']:
        print(f"Locked mappings: {locked_info['locked_persons']}")
    
    if id_changes == 0:
        print("✅ No ID changes detected - Consistency maintained!")
    else:
        print(f"⚠️  {id_changes} ID changes detected - Need improvement")
    
    if stats['total_persons'] <= 3:
        print("✅ Correct number of persons detected!")
    else:
        print(f"⚠️  Too many persons detected: {stats['total_persons']} (expected ≤3)")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_clean_tracking_visual() 