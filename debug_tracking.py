#!/usr/bin/env python3
"""
Debug script to identify tracking issues.
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


def debug_tracking():
    """Debug tracking issues step by step."""
    
    # Setup logging with DEBUG level
    setup_logger("DEBUG", "logs/debug_tracking.log")
    
    # Load configuration
    config = ConfigLoader("config/config.yaml")
    
    # Initialize components
    detector = PersonDetector(config)
    embedding_extractor = EmbeddingExtractor(config)
    reid_matcher = ReIDMatcher(config, embedding_extractor)
    tracker = PersistentTracker(config, reid_matcher)
    
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
    
    print(f"Debugging tracking on {total_frames} frames at {fps} FPS")
    print("=" * 60)
    
    # Track detailed information
    frame_details = []
    id_history = {}  # track_id -> list of person_ids over time
    
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
        
        # Collect detailed information
        frame_info = {
            'frame': frame_count,
            'timestamp': timestamp,
            'detections': len(detections),
            'persons': len(persons),
            'active_tracks': len([t for t in tracking_result.tracks if t.is_active]),
            'new_tracks': len(tracking_result.new_tracks),
            'lost_tracks': len(tracking_result.lost_tracks),
            'track_details': []
        }
        
        # Track ID changes
        for track in tracking_result.tracks:
            if track.track_id not in id_history:
                id_history[track.track_id] = []
            
            # Check if ID changed
            if id_history[track.track_id] and id_history[track.track_id][-1] != track.person_id:
                print(f"🚨 ID CHANGE at frame {frame_count}: Track {track.track_id} "
                      f"changed from {id_history[track.track_id][-1]} to {track.person_id}")
            
            id_history[track.track_id].append(track.person_id)
            
            # Get locked ID info
            locked_info = tracker.get_locked_ids_info()
            is_locked = track.track_id in locked_info['locked_persons']
            confidence = locked_info['person_confidence'].get(track.track_id, 0.0)
            
            track_detail = {
                'track_id': track.track_id,
                'person_id': track.person_id,
                'total_frames': track.total_frames,
                'is_active': track.is_active,
                'is_locked': is_locked,
                'confidence': confidence,
                'bbox': track.get_latest_detection().bbox if track.get_latest_detection() else None
            }
            frame_info['track_details'].append(track_detail)
        
        frame_details.append(frame_info)
        
        # Print detailed info every 50 frames
        if frame_count % 50 == 0:
            print(f"\nFrame {frame_count}:")
            print(f"  Detections: {frame_info['detections']}")
            print(f"  Active tracks: {frame_info['active_tracks']}")
            print(f"  New tracks: {frame_info['new_tracks']}")
            print(f"  Lost tracks: {frame_info['lost_tracks']}")
            
            # Print track details
            for track_detail in frame_info['track_details']:
                lock_status = "🔒" if track_detail['is_locked'] else "🔓"
                print(f"    Track {track_detail['track_id']} -> Person {track_detail['person_id']} "
                      f"({track_detail['total_frames']} frames, conf: {track_detail['confidence']:.3f}) {lock_status}")
            
            # Print locked IDs info
            locked_info = tracker.get_locked_ids_info()
            if locked_info['locked_persons']:
                print(f"  Locked IDs: {locked_info['locked_persons']}")
        
        frame_count += 1
        
        # Limit processing for debugging
        if frame_count >= 300:  # Process first 300 frames
            break
    
    # End session
    tracker.end_session(time.time())
    
    # Print final analysis
    print("\n" + "=" * 60)
    print("FINAL ANALYSIS")
    print("=" * 60)
    
    # Analyze ID consistency
    print("\nID Consistency Analysis:")
    for track_id, person_ids in id_history.items():
        unique_ids = set(person_ids)
        if len(unique_ids) > 1:
            print(f"  Track {track_id}: {len(unique_ids)} different IDs {unique_ids}")
        else:
            print(f"  Track {track_id}: Consistent ID {list(unique_ids)[0]}")
    
    # Count unique persons
    all_person_ids = set()
    for person_ids in id_history.values():
        all_person_ids.update(person_ids)
    
    print(f"\nTotal unique person IDs: {len(all_person_ids)}")
    print(f"Person IDs used: {sorted(all_person_ids)}")
    
    # Analyze embedding similarities
    print("\nEmbedding Analysis:")
    if len(frame_details) > 1:
        # Compare embeddings between consecutive frames
        for i in range(1, min(5, len(frame_details))):  # First 5 frame transitions
            prev_frame = frame_details[i-1]
            curr_frame = frame_details[i]
            
            if prev_frame['track_details'] and curr_frame['track_details']:
                print(f"  Frame {i-1} -> {i}: {len(prev_frame['track_details'])} -> {len(curr_frame['track_details'])} tracks")
    
    # Print configuration
    print("\nConfiguration:")
    tracking_config = config.get_tracking_config()
    print(f"  reid_threshold: {tracking_config.get('reid_threshold')}")
    print(f"  history_threshold: {tracking_config.get('history_threshold')}")
    print(f"  id_lock_threshold: {tracking_config.get('id_lock_threshold')}")
    print(f"  spatial_threshold: {tracking_config.get('spatial_threshold')}")
    print(f"  min_track_frames: {tracking_config.get('min_track_frames')}")
    
    # Cleanup
    cap.release()
    
    return frame_details, id_history


if __name__ == "__main__":
    debug_tracking() 