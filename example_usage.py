#!/usr/bin/env python3
"""
Example usage of the Person Re-identification System.

This script demonstrates how to use the system to process videos
and perform person re-identification for labor activity analysis.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from main import PersonReIDSystem
from src.utils.config_loader import ConfigLoader


def example_basic_usage():
    """Example of basic system usage."""
    print("=== Person Re-identification System - Basic Usage ===")
    
    # Initialize the system
    system = PersonReIDSystem("config/config.yaml")
    
    # Process a video
    input_video = "data/raw/video.mp4"
    output_video = "data/processed/output_reid.mp4"
    
    if os.path.exists(input_video):
        print(f"Processing video: {input_video}")
        stats = system.process_video(input_video, output_video)
        
        print("\nProcessing Results:")
        print(f"- Total frames processed: {stats['processed_frames']}")
        print(f"- Total detections: {stats['total_detections']}")
        print(f"- Total tracks: {stats['total_tracks']}")
        print(f"- Active tracks: {stats['active_tracks']}")
        print(f"- Processing time: {stats['processing_time']:.2f} seconds")
        print(f"- Processing FPS: {stats['fps_processing']:.2f}")
        
        # Save results
        system.save_results("data/processed/results", stats)
        print(f"\nResults saved to: data/processed/results")
    else:
        print(f"Input video not found: {input_video}")


def example_custom_config():
    """Example with custom configuration."""
    print("\n=== Person Re-identification System - Custom Configuration ===")
    
    # Load configuration
    config = ConfigLoader("config/config.yaml")
    
    # Modify some parameters
    config.config['detection']['confidence_threshold'] = 0.6
    config.config['reid']['similarity_threshold'] = 0.8
    config.config['tracking']['max_disappeared'] = 50
    
    # Initialize system with custom config
    system = PersonReIDSystem("config/config.yaml")
    
    # Process with custom settings
    input_video = "data/raw/video_1.mp4"
    output_video = "data/processed/output_custom.mp4"
    
    if os.path.exists(input_video):
        print(f"Processing video with custom settings: {input_video}")
        stats = system.process_video(input_video, output_video)
        
        print(f"Custom processing completed. Output: {output_video}")
    else:
        print(f"Input video not found: {input_video}")


def example_batch_processing():
    """Example of batch processing multiple videos."""
    print("\n=== Person Re-identification System - Batch Processing ===")
    
    # Initialize system
    system = PersonReIDSystem("config/config.yaml")
    
    # Get all video files in raw directory
    raw_dir = Path("data/raw")
    video_files = list(raw_dir.glob("*.mp4"))
    
    if not video_files:
        print("No video files found in data/raw/")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video
    for i, video_file in enumerate(video_files):
        print(f"\nProcessing video {i+1}/{len(video_files)}: {video_file.name}")
        
        output_file = f"data/processed/batch_output_{i+1}.mp4"
        
        try:
            stats = system.process_video(str(video_file), output_file)
            print(f"✓ Completed: {output_file}")
            print(f"  - Tracks: {stats['total_tracks']}, Active: {stats['active_tracks']}")
        except Exception as e:
            print(f"✗ Error processing {video_file.name}: {e}")
    
    print("\nBatch processing completed!")


def example_analysis():
    """Example of analyzing tracking results."""
    print("\n=== Person Re-identification System - Analysis ===")
    
    # Initialize system
    system = PersonReIDSystem("config/config.yaml")
    
    # Process a video first
    input_video = "data/raw/video.mp4"
    output_video = "data/processed/analysis_output.mp4"
    
    if not os.path.exists(input_video):
        print(f"Input video not found: {input_video}")
        return
    
    print("Processing video for analysis...")
    stats = system.process_video(input_video, output_video)
    
    # Analyze results
    print("\n=== Analysis Results ===")
    
    # Track statistics
    track_stats = system.track_manager.get_track_statistics()
    print(f"Track Analysis:")
    print(f"- Total tracks created: {track_stats['total_tracks']}")
    print(f"- Average track length: {track_stats['avg_track_length']:.1f} frames")
    print(f"- Average track duration: {track_stats['avg_track_duration']:.2f} seconds")
    print(f"- Longest track: {track_stats['max_track_length']} frames")
    
    # Person activity analysis
    print(f"\nPerson Activity Analysis:")
    print(f"- Total persons detected: {stats['total_detections']}")
    print(f"- Average detections per frame: {stats['total_detections']/stats['processed_frames']:.2f}")
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"- Processing speed: {stats['fps_processing']:.2f} FPS")
    print(f"- Total processing time: {stats['processing_time']:.2f} seconds")
    
    # Save detailed analysis
    analysis_results = {
        'track_analysis': track_stats,
        'performance_metrics': {
            'fps_processing': stats['fps_processing'],
            'total_time': stats['processing_time'],
            'frames_processed': stats['processed_frames']
        },
        'detection_metrics': {
            'total_detections': stats['total_detections'],
            'avg_detections_per_frame': stats['total_detections']/stats['processed_frames']
        }
    }
    
    import json
    with open("data/processed/analysis_results.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nDetailed analysis saved to: data/processed/analysis_results.json")


def main():
    """Main function to run examples."""
    print("Person Re-identification System - Example Usage")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Run examples
    try:
        example_basic_usage()
        example_custom_config()
        example_batch_processing()
        example_analysis()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nGenerated files:")
        print("- data/processed/output_reid.mp4 (basic processing)")
        print("- data/processed/output_custom.mp4 (custom config)")
        print("- data/processed/batch_output_*.mp4 (batch processing)")
        print("- data/processed/analysis_output.mp4 (analysis)")
        print("- data/processed/results/ (processing results)")
        print("- data/processed/analysis_results.json (detailed analysis)")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 