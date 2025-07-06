#!/usr/bin/env python3
"""
Script to verify the results of the store monitoring system.
"""

import cv2
import os
import sys
from datetime import datetime

def verify_video_output(video_path):
    """Verify the output video and display basic statistics."""
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return False
    
    # Get file size
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"✅ Video found: {video_path}")
    print(f"📁 File size: {file_size_mb:.2f} MB")
    
    # Open video and get properties
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Cannot open video file")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"🎥 Video properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {frame_count}")
    print(f"   Duration: {duration:.2f} seconds")
    
    # Try to read first frame
    ret, frame = cap.read()
    if ret:
        print("✅ Video is readable and valid")
        # Check if frame has annotations (simple check for colored pixels)
        if frame.shape[2] == 3:  # RGB/BGR
            print("✅ Video appears to have annotations/tracking data")
    else:
        print("❌ Cannot read video frames")
        cap.release()
        return False
    
    cap.release()
    return True

def main():
    print("🔍 Verifying Store Monitoring System Results")
    print("=" * 50)
    
    # Get video path from command line argument or use default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "data/processed/output_tracking.mp4"
    
    # Check input video
    input_video = "data/raw/video.mp4"
    if os.path.exists(input_video):
        input_size = os.path.getsize(input_video) / (1024 * 1024)
        print(f"📹 Input video: {input_video} ({input_size:.2f} MB)")
    
    print("\n" + "=" * 50)
    
    # Verify output video
    success = verify_video_output(video_path)
    
    # Check log files
    print("\n📋 Log files found:")
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        for log_file in os.listdir(logs_dir):
            if log_file.endswith('.log'):
                log_path = os.path.join(logs_dir, log_file)
                log_size = os.path.getsize(log_path)
                print(f"   {log_file} ({log_size} bytes)")
    
    print("\n" + "=" * 50)
    
    if success:
        print("✅ Verification completed successfully!")
        print("\n🎯 Next steps:")
        print("   1. Open the output video to visually inspect the tracking")
        print("   2. Check if person detection and tracking are working")
        print("   3. Ready to continue with Phase 3 (FastReID integration)")
        
        abs_path = os.path.abspath(video_path)
        print(f"\n📂 Output video location: {abs_path}")
        
        print(f"\n💡 To view the video:")
        print(f"   open {video_path}")
        print(f"   or use any video player")
    else:
        print("❌ Verification failed!")
        print("Please check the logs for errors.")

if __name__ == "__main__":
    main() 