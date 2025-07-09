"""
Video writer for person re-identification system.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging
from pathlib import Path

from ..utils.config_loader import ConfigLoader


class VideoWriter:
    """Video writer with tracking visualization."""
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize video writer.
        
        Args:
            config: Configuration loader
        """
        self.config = config
        self.video_config = config.get_video_config()
        
        # Video parameters
        self.output_fps = self.video_config.get("output_fps", 30)
        self.output_codec = self.video_config.get("output_codec", "mp4v")
        self.output_quality = self.video_config.get("output_quality", 95)
        self.resize_factor = self.video_config.get("resize_factor", 1.0)
        
        self.writer = None
        self.output_path = None
        self.frame_count = 0
        
        logging.info("VideoWriter initialized")
    
    def open(self, output_path: str, width: int, height: int, fps: Optional[float] = None):
        """
        Open video writer.
        
        Args:
            output_path: Output video path
            width: Video width
            height: Video height
            fps: Video FPS (optional, uses config default if None)
        """
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Apply resize factor
        output_width = int(width * self.resize_factor)
        output_height = int(height * self.resize_factor)
        
        # Use provided fps or config default
        fps = fps if fps is not None else self.output_fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.output_codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        if not self.writer.isOpened():
            raise ValueError(f"Could not create video writer: {output_path}")
        
        self.output_path = output_path
        self.frame_count = 0
        
        logging.info(f"Video writer opened: {output_path} ({output_width}x{output_height}, {fps} FPS)")
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the video.
        
        Args:
            frame: Frame to write
        """
        if self.writer is None:
            raise RuntimeError("Video writer not opened. Call open() first.")
        
        # Resize frame if needed
        if self.resize_factor != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * self.resize_factor)
            new_height = int(height * self.resize_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def close(self):
        """Close video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
            logging.info(f"Video writer closed: {self.output_path} ({self.frame_count} frames)")
    
    def get_frame_count(self) -> int:
        """Get number of frames written."""
        return self.frame_count
    
    def is_opened(self) -> bool:
        """Check if video writer is opened."""
        return self.writer is not None and self.writer.isOpened()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 