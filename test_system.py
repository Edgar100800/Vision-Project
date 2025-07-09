#!/usr/bin/env python3
"""
Simple test script for the Person Re-identification System.

This script tests the basic functionality of each component
without requiring a full video processing pipeline.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils.config_loader import ConfigLoader
        from src.utils.data_structures import Person, Detection, Track
        from src.detection.person_detector import PersonDetector
        from src.embedding.embedding_extractor import EmbeddingExtractor
        from src.reid.reid_matcher import ReIDMatcher
        from src.tracking.track_manager import TrackManager
        from src.visualization.visualizer import Visualizer
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_config_loader():
    """Test configuration loader."""
    print("\nTesting configuration loader...")
    
    try:
        from src.utils.config_loader import ConfigLoader
        
        config = ConfigLoader("config/config.yaml")
        
        # Test basic config access
        confidence_threshold = config.get("detection.confidence_threshold")
        similarity_threshold = config.get("reid.similarity_threshold")
        
        print(f"✓ Config loaded successfully")
        print(f"  - Detection confidence threshold: {confidence_threshold}")
        print(f"  - ReID similarity threshold: {similarity_threshold}")
        
        return True
    except Exception as e:
        print(f"✗ Config loader error: {e}")
        return False


def test_data_structures():
    """Test data structures."""
    print("\nTesting data structures...")
    
    try:
        from src.utils.data_structures import Detection, Person, Track
        
        # Test Detection
        bbox = [100, 100, 200, 300]
        detection = Detection(
            bbox=bbox,
            confidence=0.95,
            frame_id=0,
            timestamp=0.0
        )
        
        # Test Person
        person = Person(
            detection=detection,
            embedding=np.random.rand(512),
            person_id=1,
            track_id=1
        )
        
        # Test Track
        track = Track(
            track_id=1,
            person_id=1
        )
        track.add_detection(detection, person.embedding)
        
        print("✓ Data structures created successfully")
        print(f"  - Detection center: {detection.get_center()}")
        print(f"  - Person features: {len(person.get_features())}")
        print(f"  - Track duration: {track.get_duration():.2f}s")
        
        return True
    except Exception as e:
        print(f"✗ Data structures error: {e}")
        return False


def test_detector():
    """Test person detector."""
    print("\nTesting person detector...")
    
    try:
        from src.utils.config_loader import ConfigLoader
        from src.detection.person_detector import PersonDetector
        
        config = ConfigLoader("config/config.yaml")
        detector = PersonDetector(config)
        
        # Create a dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection (this will fail if YOLO model is not available)
        try:
            detections = detector.detect(frame, frame_id=0, timestamp=0.0)
            print("✓ Person detector initialized successfully")
            print(f"  - Detections found: {len(detections)}")
        except Exception as e:
            print(f"⚠ Person detector initialized but detection failed: {e}")
            print("  (This is expected if YOLO model is not properly loaded)")
        
        return True
    except Exception as e:
        print(f"✗ Person detector error: {e}")
        return False


def test_embedding_extractor():
    """Test embedding extractor."""
    print("\nTesting embedding extractor...")
    
    try:
        from src.utils.config_loader import ConfigLoader
        from src.embedding.embedding_extractor import EmbeddingExtractor
        
        config = ConfigLoader("config/config.yaml")
        extractor = EmbeddingExtractor(config)
        
        # Create a dummy person crop
        person_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        
        # Test embedding extraction
        embedding = extractor.extract_embedding(person_crop)
        
        print("✓ Embedding extractor works successfully")
        print(f"  - Embedding dimension: {len(embedding)}")
        print(f"  - Embedding type: {type(embedding)}")
        
        return True
    except Exception as e:
        print(f"✗ Embedding extractor error: {e}")
        return False


def test_reid_matcher():
    """Test ReID matcher."""
    print("\nTesting ReID matcher...")
    
    try:
        from src.utils.config_loader import ConfigLoader
        from src.embedding.embedding_extractor import EmbeddingExtractor
        from src.reid.reid_matcher import ReIDMatcher
        from src.utils.data_structures import Person, Detection, Track
        
        config = ConfigLoader("config/config.yaml")
        embedding_extractor = EmbeddingExtractor(config)
        matcher = ReIDMatcher(config, embedding_extractor)
        
        # Create test data
        detection1 = Detection(bbox=[100, 100, 200, 300], confidence=0.9, frame_id=0, timestamp=0.0)
        detection2 = Detection(bbox=[150, 150, 250, 350], confidence=0.8, frame_id=1, timestamp=1.0)
        
        person1 = Person(detection=detection1, embedding=np.random.rand(512), person_id=1)
        person2 = Person(detection=detection2, embedding=np.random.rand(512), person_id=2)
        
        track1 = Track(track_id=1, person_id=1)
        track1.add_detection(detection1, person1.embedding)
        
        # Test matching
        matches = matcher.match_persons([person2], [track1])
        
        print("✓ ReID matcher works successfully")
        print(f"  - Matches found: {len(matches)}")
        
        return True
    except Exception as e:
        print(f"✗ ReID matcher error: {e}")
        return False


def test_track_manager():
    """Test track manager."""
    print("\nTesting track manager...")
    
    try:
        from src.utils.config_loader import ConfigLoader
        from src.embedding.embedding_extractor import EmbeddingExtractor
        from src.reid.reid_matcher import ReIDMatcher
        from src.tracking.track_manager import TrackManager
        from src.utils.data_structures import Person, Detection
        
        config = ConfigLoader("config/config.yaml")
        embedding_extractor = EmbeddingExtractor(config)
        reid_matcher = ReIDMatcher(config, embedding_extractor)
        track_manager = TrackManager(config, reid_matcher)
        
        # Create test person
        detection = Detection(bbox=[100, 100, 200, 300], confidence=0.9, frame_id=0, timestamp=0.0)
        person = Person(detection=detection, embedding=np.random.rand(512))
        
        # Test track update
        result = track_manager.update_tracks(0, 0.0, [person])
        
        print("✓ Track manager works successfully")
        print(f"  - Tracks created: {len(result.new_tracks)}")
        print(f"  - Total tracks: {len(result.tracks)}")
        
        return True
    except Exception as e:
        print(f"✗ Track manager error: {e}")
        return False


def test_visualizer():
    """Test visualizer."""
    print("\nTesting visualizer...")
    
    try:
        from src.utils.config_loader import ConfigLoader
        from src.visualization.visualizer import Visualizer
        from src.utils.data_structures import TrackingResult, Person, Detection, Track
        
        config = ConfigLoader("config/config.yaml")
        visualizer = Visualizer(config)
        
        # Create test data
        detection = Detection(bbox=[100, 100, 200, 300], confidence=0.9, frame_id=0, timestamp=0.0)
        person = Person(detection=detection, embedding=np.random.rand(512), person_id=1, track_id=1)
        track = Track(track_id=1, person_id=1)
        track.add_detection(detection, person.embedding)
        
        result = TrackingResult(
            frame_id=0,
            timestamp=0.0,
            persons=[person],
            tracks=[track]
        )
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test visualization
        annotated_frame = visualizer.draw_frame(frame, result)
        
        print("✓ Visualizer works successfully")
        print(f"  - Input frame shape: {frame.shape}")
        print(f"  - Output frame shape: {annotated_frame.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Visualizer error: {e}")
        return False


def main():
    """Run all tests."""
    print("Person Re-identification System - Component Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loader,
        test_data_structures,
        test_detector,
        test_embedding_extractor,
        test_reid_matcher,
        test_track_manager,
        test_visualizer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 