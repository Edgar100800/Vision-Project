#!/usr/bin/env python3
"""
Command-line script to run the Person Re-identification System
Usage: python run_demo.py --video data/raw/video.mp4
"""

from config.global_config import get_config, apply_preset, config
from src.main_pipeline import PersonReIDPipeline
from src.campaign_pipeline import CampaignPersonReIDPipeline
import argparse
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.append('src')

# Import configuration


def main():
    parser = argparse.ArgumentParser(
        description="Person Re-identification System Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--video",
        default="data/raw/video.mp4",
        help="Path to input video file"
    )

    parser.add_argument(
        "--output",
        default="output/demo_results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--preset",
        choices=["high_accuracy", "high_speed",
                 "balanced", "three_person_dataset"],
        help="Configuration preset to use"
    )

    parser.add_argument(
        "--yolo-model",
        help="YOLO model path (uses config default if not specified)"
    )

    parser.add_argument(
        "--reid-model",
        help="ReID model name (uses config default if not specified)"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames to process (for testing)"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference (uses config default if not specified)"
    )

    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Don't save person crops"
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Don't save annotated video"
    )

    parser.add_argument(
        "--hnsw-dim",
        type=int,
        help="HNSW index dimension (uses config default if not specified)"
    )

    parser.add_argument(
        "--hnsw-max-elements",
        type=int,
        help="HNSW max elements (uses config default if not specified)"
    )

    parser.add_argument(
        "--config-summary",
        action="store_true",
        help="Show configuration summary and exit"
    )

    parser.add_argument(
        "--enable-campaigns",
        action="store_true",
        help="Enable campaign-based ID assignment (delayed IDs)"
    )

    parser.add_argument(
        "--campaign-frames",
        type=int,
        default=25,
        help="Frames required for campaign completion"
    )

    parser.add_argument(
        "--campaign-samples",
        type=int,
        default=6,
        help="Minimum samples required for ID assignment"
    )

    args = parser.parse_args()

    # Show configuration summary if requested
    if args.config_summary:
        config.print_config_summary()
        sys.exit(0)

    # Apply preset if specified
    if args.preset:
        print(f"🔧 Applying configuration preset: {args.preset}")
        apply_preset(args.preset)

    # Enable campaigns for three_person_dataset preset or if explicitly requested
    enable_campaigns = args.enable_campaigns or args.preset == 'three_person_dataset'

    # Get current configuration values
    detection_config = get_config('detection')
    reid_config = get_config('reid')
    indexing_config = get_config('indexing')
    output_config = get_config('output')

    # Use config defaults for unspecified arguments
    yolo_model = args.yolo_model or detection_config['model_path']
    reid_model = args.reid_model or reid_config['model_name']
    device = args.device or reid_config['device']
    hnsw_dim = args.hnsw_dim or indexing_config['dimension']
    hnsw_max_elements = args.hnsw_max_elements or indexing_config['max_elements']

    # Validate input video
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found at {args.video}")
        print("Please provide a valid video file path.")
        sys.exit(1)

    print("=" * 80)
    print("PERSON RE-IDENTIFICATION SYSTEM")
    print("=" * 80)
    print(f"Input video: {args.video}")
    print(f"Output directory: {args.output}")
    print(f"YOLO model: {yolo_model}")
    print(f"ReID model: {reid_model}")
    print(f"Device: {device}")
    print(f"HNSW dimension: {hnsw_dim}")
    print(f"HNSW max elements: {hnsw_max_elements}")
    if args.preset:
        print(f"Configuration preset: {args.preset}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames}")
    if enable_campaigns:
        print(f"Campaign system: ENABLED")
        print(f"Campaign frames: {args.campaign_frames}")
        print(f"Campaign samples: {args.campaign_samples}")
    else:
        print(f"Campaign system: DISABLED")
    print("=" * 80)

    try:
        # Initialize pipeline
        print("\n🚀 Initializing pipeline...")

        if enable_campaigns:
            pipeline = CampaignPersonReIDPipeline(
                enable_campaigns=True,
                campaign_frames=args.campaign_frames,
                campaign_min_samples=args.campaign_samples,
                campaign_confidence_threshold=0.75,
                yolo_model_path=yolo_model,
                reid_model_name=reid_model,
                hnsw_dim=hnsw_dim,
                hnsw_max_elements=hnsw_max_elements,
                output_dir=args.output,
                save_crops=not args.no_crops,
                save_video=not args.no_video,
                device=device,
                config_preset=args.preset
            )
            print("✓ Campaign pipeline initialized successfully")
        else:
            pipeline = PersonReIDPipeline(
                yolo_model_path=yolo_model,
                reid_model_name=reid_model,
                hnsw_dim=hnsw_dim,
                hnsw_max_elements=hnsw_max_elements,
                output_dir=args.output,
                save_crops=not args.no_crops,
                save_video=not args.no_video,
                device=device,
                config_preset=args.preset
            )
            print("✓ Standard pipeline initialized successfully")

        # Process video
        print("\n📹 Processing video...")
        start_time = time.time()

        stats = pipeline.process_video(
            video_path=args.video,
            max_frames=args.max_frames
        )

        processing_time = time.time() - start_time

        # Display results
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED")
        print("=" * 80)
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"📊 Total frames: {stats['total_frames']}")
        print(f"🔍 Total detections: {stats['total_detections']}")
        print(f"📈 Average FPS: {stats['avg_fps']:.2f}")

        if enable_campaigns:
            # Campaign statistics
            campaign_stats = stats.get('campaign_stats', {})
            print(
                f"👥 Total confirmations: {stats.get('total_confirmations', 0)}")
            print(
                f"🎯 Active candidates: {campaign_stats.get('active_candidates', 0)}")
            print(
                f"✅ Confirmed persons: {campaign_stats.get('confirmed_persons', 0)}")
            print(
                f"📈 Campaign success rate: {campaign_stats.get('success_rate', 0):.2%}")

            # Campaign details
            print(f"\n🎯 Campaign Statistics:")
            print(
                f"  - Total campaigns: {campaign_stats.get('total_campaigns', 0)}")
            print(
                f"  - Successful campaigns: {campaign_stats.get('successful_campaigns', 0)}")
            print(
                f"  - Failed campaigns: {campaign_stats.get('failed_campaigns', 0)}")

            # Show confirmed persons
            confirmed_persons = pipeline.get_confirmed_persons()
            if confirmed_persons:
                print(f"\n👥 Confirmed Persons:")
                for person in confirmed_persons:
                    print(
                        f"  - Person {person['person_id']}: {person['total_detections']} detections")
                    print(f"    Cluster: {person['cluster_id']}")
                    print(
                        f"    Confirmed at frame: {person['confirmation_frame']}")
        else:
            # Standard statistics
            fusion_stats = stats.get('fusion_stats', {})
            print(f"👥 Unique persons: {fusion_stats.get('next_global_id', 0)}")
            print(f"🎯 Active persons: {fusion_stats.get('active_persons', 0)}")

            # Performance breakdown
            processing_times = stats.get('processing_times', {})
            if isinstance(processing_times, dict):
                print(f"\n📊 Performance Breakdown:")
                for stage, times in processing_times.items():
                    if stage != 'total' and isinstance(times, dict):
                        print(
                            f"  - {stage.capitalize()}: {times.get('mean', 0)*1000:.1f}ms")

            # Fusion statistics
            if fusion_stats:
                print(f"\n🔗 ID Fusion Statistics:")
                print(
                    f"  - Total fusions: {fusion_stats.get('total_fusions', 0)}")
                print(
                    f"  - ReID matches: {fusion_stats.get('reid_matches', 0)}")
                print(
                    f"  - Tracking matches: {fusion_stats.get('tracking_matches', 0)}")
                print(
                    f"  - New identities: {fusion_stats.get('new_identities', 0)}")

        # Output files
        output_path = Path(args.output)
        print(f"\n📁 Output files:")

        if not args.no_video:
            video_dir = output_path / "videos"
            if video_dir.exists():
                video_files = list(video_dir.glob("*.mp4"))
                if video_files:
                    print(f"  - Annotated video: {video_files[0]}")

        if not args.no_crops:
            crops_dir = output_path / "crops"
            if crops_dir.exists():
                person_dirs = list(crops_dir.glob("person_*"))
                if person_dirs:
                    total_crops = sum(len(list(d.glob("*.jpg")))
                                      for d in person_dirs)
                    print(
                        f"  - Person crops: {total_crops} images in {len(person_dirs)} folders")

        data_dir = output_path / "data"
        if data_dir.exists():
            print(f"  - Processing data: {data_dir}")

        log_file = output_path / "pipeline.log"
        if log_file.exists():
            print(f"  - System logs: {log_file}")

        print("\n✅ Processing completed successfully!")
        print("=" * 80)

        # Show person galleries summary
        if not enable_campaigns:
            fusion_stats = stats.get('fusion_stats', {})
            next_global_id = fusion_stats.get('next_global_id', 0)
            if next_global_id > 0:
                print(f"\n👥 Person Galleries:")
                for person_id in range(min(5, next_global_id)):
                    crop_paths = pipeline.get_person_gallery(
                        person_id, max_crops=50)
                    if crop_paths:
                        print(
                            f"  - Person {person_id}: {len(crop_paths)} crops")

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
