#!/usr/bin/env python3
"""
Test script for ID consistency with video.mp4.
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


def test_id_consistency():
    """Test ID consistency with video.mp4."""
    
    # Setup logging
    setup_logger("INFO", "logs/test_id_consistency.log")
    
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
    
    print(f"Testing ID consistency on {total_frames} frames at {fps} FPS")
    print("Status colors:")
    print("  GREEN: NEW person")
    print("  ORANGE: TRACKING existing person")
    print("  BLUE: TRACKING_LOCKED (ID locked)")
    print("  MAGENTA: RE_IDENTIFIED person from history")
    print("  GRAY: LOST person")
    
    # Track ID consistency
    id_changes = {}  # person_id -> list of frame changes
    current_ids = {}  # track_id -> person_id mapping
    
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
        
        # Check for ID changes
        for track in tracking_result.tracks:
            if track.track_id in current_ids:
                old_person_id = current_ids[track.track_id]
                if old_person_id != track.person_id:
                    # ID changed!
                    if track.person_id not in id_changes:
                        id_changes[track.person_id] = []
                    id_changes[track.person_id].append({
                        'frame': frame_count,
                        'old_id': old_person_id,
                        'new_id': track.person_id,
                        'timestamp': timestamp
                    })
                    print(f"⚠️  ID CHANGE at frame {frame_count}: Track {track.track_id} "
                          f"changed from Person {old_person_id} to Person {track.person_id}")
            current_ids[track.track_id] = track.person_id
        
        # Update person status
        for person in persons:
            if person.person_id is not None:
                if any(track.person_id == person.person_id for track in tracking_result.new_tracks):
                    person.status = "NEW"
                elif any(track.person_id == person.person_id for track in tracking_result.tracks if track.total_frames > 1):
                    person.status = "TRACKING"
                else:
                    person.status = "RE_IDENTIFIED"
        
        # Visualize results
        annotated_frame = visualizer.draw_frame(frame, tracking_result)
        
        # Add status legend
        legend_text = [
            "Status Legend:",
            "GREEN: NEW person",
            "ORANGE: TRACKING existing",
            "BLUE: TRACKING_LOCKED (ID locked)",
            "MAGENTA: RE_IDENTIFIED",
            "GRAY: LOST person"
        ]
        
        y_offset = 150
        for i, text in enumerate(legend_text):
            cv2.putText(annotated_frame, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
            cv2.putText(annotated_frame, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1)
        
        # Add locked IDs info
        locked_info = tracker.get_locked_ids_info()
        if locked_info['locked_persons']:
            locked_text = f"Locked IDs: {len(locked_info['locked_persons'])}"
            cv2.putText(annotated_frame, locked_text, (10, y_offset + 175), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 255, 255], 2)
        
        # Show frame
        cv2.imshow("ID Consistency Test", annotated_frame)
        
        # Print statistics every 100 frames
        if frame_count % 100 == 0:
            stats = tracker.get_track_statistics()
            locked_info = tracker.get_locked_ids_info()
            print(f"Frame {frame_count}: Active tracks: {stats['active_tracks']}, "
                  f"Total persons: {stats['total_persons']}, "
                  f"Locked IDs: {len(locked_info['locked_persons'])}")
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            cv2.imwrite(f"data/processed/consistency_frame_{frame_count:04d}.jpg", annotated_frame)
            print(f"Saved frame {frame_count}")
        
        frame_count += 1
        
        # Limit processing for testing
        if frame_count >= 1000:  # Process first 1000 frames
            break
    
    # End session
    tracker.end_session(time.time())
    
    # Print final statistics
    stats = tracker.get_track_statistics()
    locked_info = tracker.get_locked_ids_info()
    
    print("\n=== Final Statistics ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Active tracks: {stats['active_tracks']}")
    print(f"Total unique persons: {stats['total_persons']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"ReID matches: {stats['total_reid_matches']}")
    print(f"Locked IDs: {len(locked_info['locked_persons'])}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    
    # Print ID changes
    if id_changes:
        print("\n⚠️  ID CHANGES DETECTED:")
        for person_id, changes in id_changes.items():
            print(f"Person {person_id} had {len(changes)} ID changes:")
            for change in changes:
                print(f"  Frame {change['frame']}: {change['old_id']} -> {change['new_id']}")
    else:
        print("\n✅ No ID changes detected - Consistency maintained!")
    
    # Export history
    tracker.export_history_json("data/processed/consistency_history.json")
    print("History exported to data/processed/consistency_history.json")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_id_consistency() 