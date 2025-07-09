"""
Video processing utilities for the person re-identification system.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Generator
import logging

from .config_loader import ConfigLoader
from .data_structures import VideoMetadata


class VideoProcessor:
    """Video processing utilities."""
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize video processor.
        
        Args:
            config: Configuration loader
        """
        self.config = config
        self.video_config = config.get_video_config()
        
        # Video processing parameters
        self.output_fps = self.video_config.get("output_fps", 30)
        self.output_codec = self.video_config.get("output_codec", "mp4v")
        self.output_quality = self.video_config.get("output_quality", 95)
        self.resize_factor = self.video_config.get("resize_factor", 1.0)
    
    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Get codec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec,
            filename=video_path
        )
    
    def create_video_writer(self, output_path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
        """
        Create video writer.
        
        Args:
            output_path: Output video path
            width: Video width
            height: Video height
            fps: Video FPS
            
        Returns:
            Video writer
        """
        # Apply resize factor
        output_width = int(width * self.resize_factor)
        output_height = int(height * self.resize_factor)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.output_codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        if not writer.isOpened():
            raise ValueError(f"Could not create video writer: {output_path}")
        
        return writer
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame according to configuration.
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        if self.resize_factor != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * self.resize_factor)
            new_height = int(height * self.resize_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    def frame_generator(self, video_path: str, max_frames: Optional[int] = None) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Generate frames from video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (optional)
            
        Yields:
            Tuple of (frame, frame_id, timestamp)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_id = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_id / fps if fps > 0 else 0
                yield frame, frame_id, timestamp
                
                frame_id += 1
                
                if max_frames and frame_id >= max_frames:
                    break
        finally:
            cap.release()
    
    def extract_frames(self, video_path: str, output_dir: str, frame_interval: int = 1) -> int:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for frames
            frame_interval: Extract every Nth frame
            
        Returns:
            Number of extracted frames
        """
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extracted_count = 0
        
        for frame, frame_id, timestamp in self.frame_generator(video_path):
            if frame_id % frame_interval == 0:
                frame_filename = f"frame_{frame_id:06d}.jpg"
                frame_path = output_path / frame_filename
                cv2.imwrite(str(frame_path), frame)
                extracted_count += 1
        
        logging.info(f"Extracted {extracted_count} frames to {output_dir}")
        return extracted_count
    
    def create_video_from_frames(self, frame_dir: str, output_path: str, fps: float = 30) -> bool:
        """
        Create video from frame images.
        
        Args:
            frame_dir: Directory containing frame images
            output_path: Output video path
            fps: Video FPS
            
        Returns:
            True if successful
        """
        import glob
        from pathlib import Path
        
        frame_path = Path(frame_dir)
        frame_files = sorted(glob.glob(str(frame_path / "*.jpg")))
        
        if not frame_files:
            logging.error(f"No frame files found in {frame_dir}")
            return False
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        height, width = first_frame.shape[:2]
        
        # Create video writer
        writer = self.create_video_writer(output_path, width, height, fps)
        
        try:
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                if frame is not None:
                    writer.write(frame)
        finally:
            writer.release()
        
        logging.info(f"Created video: {output_path}")
        return True
    
    def get_frame_at_time(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """
        Get frame at specific timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds
            
        Returns:
            Frame at timestamp or None if not found
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        cap.release()
        
        return frame if ret else None 