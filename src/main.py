"""
Main script for the store monitoring system.
Basic pipeline: YOLO detection + Deep SORT tracking.
"""

import argparse
import cv2
import time
import os
import sys
from typing import Optional

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import get_config
from utils.logger import setup_logger
from detectors.yolo_detector import YOLODetector
from trackers.deep_sort_tracker import DeepSORTTracker
from reid.reId import ReIDModel


def main():
    """Main function to run the store monitoring system."""
    parser = argparse.ArgumentParser(description="Store Monitoring System")
    parser.add_argument("--video", type=str,
                        default="data/video.mp4", help="Path to input video")
    parser.add_argument(
        "--output", type=str, default="data/processed/output.mp4", help="Path to output video")
    parser.add_argument("--display", action="store_true",
                        help="Display video in real-time")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--config", type=str,
                        help="Path to configuration file")

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger("StoreMonitor", "logs/main.log")
    logger.info("Starting Store Monitoring System")

    # Load configuration
    config = get_config()
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        # TODO: Implement config loading from file

    # Override video paths from arguments
    config.video.input_path = args.video
    config.video.output_path = args.output
    config.video.fps = args.fps

    # Check if input video exists
    if not os.path.exists(config.video.input_path):
        logger.error(f"Input video not found: {config.video.input_path}")
        return

    # Create output directory
    os.makedirs(os.path.dirname(config.video.output_path), exist_ok=True)

    try:
        # Initialize detector and tracker
        logger.info("Initializing YOLO detector...")
        detector = YOLODetector(config.detection)

        # --- Inicializar Re-ID Model ---
        logger.info("Initializing Re-ID model...")
        reid_model = ReIDModel(threshold=getattr(config.reid, 'threshold', 0.75)) # Usa umbral de config si existe

        logger.info("Initializing Deep SORT tracker...")
        tracker = DeepSORTTracker(config.tracking, reid_model=reid_model)

        # Open video
        logger.info(f"Opening video: {config.video.input_path}")
        cap = cv2.VideoCapture(config.video.input_path)

        if not cap.isOpened():
            logger.error("Failed to open video")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(config.video.output_path,
                              fourcc, config.video.fps, (width, height))

        # Processing loop
        frame_count = 0
        start_time = time.time()

        logger.info("Starting video processing...")

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

            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)} | Tracks: {len(tracks)}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write frame
            out.write(annotated_frame)

            # Display if requested
            if args.display:
                cv2.imshow("Store Monitor", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested exit")
                    break

            # Log progress
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                progress = (frame_count / total_frames) * 100
                logger.info(
                    f"Progress: {progress:.1f}% | FPS: {fps_current:.1f}")

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Final statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed

        logger.info(f"Processing completed!")
        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Total time: {elapsed:.2f} seconds")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Output saved to: {config.video.output_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
