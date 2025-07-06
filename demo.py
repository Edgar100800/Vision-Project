#!/usr/bin/env python3
"""
Demo script for the Store Monitoring System.
This script demonstrates the basic functionality without requiring a video file.
"""

from trackers.deep_sort_tracker import DeepSORTTracker
from detectors.yolo_detector import YOLODetector
from utils.logger import setup_logger
from utils.config import get_config
import cv2
import numpy as np
import time
import os
from typing import List, Tuple

# Add src to path
import sys
sys.path.append('src')


def create_demo_video(output_path: str = "data/demo_video.mp4", duration: int = 30):
    """
    Create a demo video with moving rectangles simulating people.

    Args:
        output_path: Path to save the demo video
        duration: Duration in seconds
    """
    print("🎬 Creating demo video...")

    # Video properties
    width, height = 1280, 720
    fps = 30
    total_frames = duration * fps

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create moving objects (simulating people)
    people = [
        {"x": 100, "y": 300, "vx": 2, "vy": 0, "w": 60, "h": 120},
        {"x": 200, "y": 400, "vx": 1, "vy": 1, "w": 50, "h": 100},
        {"x": 800, "y": 200, "vx": -1, "vy": 0.5, "w": 70, "h": 140},
        {"x": 1000, "y": 500, "vx": -2, "vy": -0.5, "w": 55, "h": 110},
    ]

    for frame_idx in range(total_frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background

        # Draw store layout
        # Entrance line
        cv2.line(frame, (100, 0), (100, height), (0, 255, 0), 3)
        cv2.putText(frame, "ENTRANCE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Exit line
        cv2.line(frame, (1180, 0), (1180, height), (0, 0, 255), 3)
        cv2.putText(frame, "EXIT", (1190, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Queue area
        cv2.rectangle(frame, (200, 400), (800, 600), (255, 255, 0), 2)
        cv2.putText(frame, "QUEUE AREA", (400, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Cashier area
        cv2.rectangle(frame, (900, 400), (1100, 600), (255, 0, 255), 2)
        cv2.putText(frame, "CASHIER", (920, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Update and draw people
        for person in people:
            # Update position
            person["x"] += person["vx"]
            person["y"] += person["vy"]

            # Bounce off walls
            if person["x"] <= 0 or person["x"] + person["w"] >= width:
                person["vx"] *= -1
            if person["y"] <= 0 or person["y"] + person["h"] >= height:
                person["vy"] *= -1

            # Keep in bounds
            person["x"] = max(0, min(person["x"], width - person["w"]))
            person["y"] = max(0, min(person["y"], height - person["h"]))

            # Draw person (rectangle)
            x, y, w, h = int(person["x"]), int(
                person["y"]), person["w"], person["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

        # Progress
        if frame_idx % (fps * 5) == 0:
            print(f"   Progress: {frame_idx / total_frames * 100:.1f}%")

    out.release()
    print(f"✅ Demo video created: {output_path}")


def run_demo():
    """Run the demo with the created video."""
    print("🚀 Running Store Monitoring Demo")
    print("=" * 50)

    # Setup
    demo_video = "data/demo_video.mp4"
    output_video = "data/processed/demo_output.mp4"

    # Create demo video if it doesn't exist
    if not os.path.exists(demo_video):
        os.makedirs("data", exist_ok=True)
        create_demo_video(demo_video)

    # Setup logger
    logger = setup_logger("Demo", "logs/demo.log")

    # Load configuration
    config = get_config()
    config.video.input_path = demo_video
    config.video.output_path = output_video

    try:
        # Initialize detector and tracker
        logger.info("Initializing YOLO detector...")
        detector = YOLODetector(config.detection)

        logger.info("Initializing Deep SORT tracker...")
        tracker = DeepSORTTracker(config.tracking)

        # Process video
        logger.info(f"Processing demo video: {demo_video}")
        cap = cv2.VideoCapture(demo_video)

        if not cap.isOpened():
            logger.error("Failed to open demo video")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        print("🎥 Processing frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run detection
            detections = detector.detect(frame)

            # Run tracking
            tracks = tracker.update(detections, frame)

            # Draw results
            annotated_frame = tracker.draw_tracks(frame, tracks)

            # Add statistics
            stats_text = [
                f"Frame: {frame_count}/{total_frames}",
                f"Detections: {len(detections)}",
                f"Active Tracks: {len(tracks)}",
                f"FPS: {frame_count / (time.time() - start_time):.1f}"
            ]

            for i, text in enumerate(stats_text):
                cv2.putText(annotated_frame, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Write frame
            out.write(annotated_frame)

            # Progress
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}%")

        # Cleanup
        cap.release()
        out.release()

        # Statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed

        print("\n📊 Demo Results")
        print("-" * 30)
        print(f"Total frames processed: {frame_count}")
        print(f"Processing time: {elapsed:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Output saved to: {output_video}")

        print("\n🎉 Demo completed successfully!")
        print("You can now:")
        print("1. View the output video")
        print("2. Try with your own video: python src/main.py --video your_video.mp4")
        print("3. Run setup: python setup.py")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    run_demo()
