#!/usr/bin/env python3
"""
Example script showing how to use the global configuration system
Demonstrates different ways to configure and use the person re-identification system
"""

from src.main_pipeline import PersonReIDPipeline
from config.global_config import get_config, update_config, apply_preset, config
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage with Default Configuration")
    print("=" * 80)

    # Get configuration for different modules
    detection_config = get_config('detection')
    reid_config = get_config('reid')

    print(
        f"Detection confidence threshold: {detection_config['confidence_threshold']}")
    print(f"ReID model: {reid_config['model_name']}")
    print(f"ReID image size: {reid_config['image_size']}")

    # Initialize pipeline with default configuration
    pipeline = PersonReIDPipeline()
    print("✓ Pipeline initialized with default configuration")


def example_2_custom_configuration():
    """Example 2: Customizing specific configuration parameters"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Customizing Specific Configuration Parameters")
    print("=" * 80)

    # Update specific configuration parameters
    update_config('detection', 'confidence_threshold', 0.7)
    update_config('reid', 'model_name', 'osnet_x0_75')
    update_config('fusion', 'similarity_threshold', 0.8)

    # Get updated configuration
    detection_config = get_config('detection')
    reid_config = get_config('reid')
    fusion_config = get_config('fusion')

    print(
        f"Updated detection confidence: {detection_config['confidence_threshold']}")
    print(f"Updated ReID model: {reid_config['model_name']}")
    print(
        f"Updated fusion similarity: {fusion_config['similarity_threshold']}")

    # Initialize pipeline with updated configuration
    pipeline = PersonReIDPipeline()
    print("✓ Pipeline initialized with custom configuration")


def example_3_configuration_presets():
    """Example 3: Using configuration presets"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Using Configuration Presets")
    print("=" * 80)

    # Show available presets
    print("Available configuration presets:")
    print("- high_accuracy: Optimized for maximum precision")
    print("- high_speed: Optimized for maximum speed")
    print("- balanced: Balance between accuracy and speed")

    # Apply high accuracy preset
    print("\nApplying 'high_accuracy' preset...")
    apply_preset('high_accuracy')

    detection_config = get_config('detection')
    reid_config = get_config('reid')
    indexing_config = get_config('indexing')

    print(f"Detection model: {detection_config['model_path']}")
    print(f"ReID model: {reid_config['model_name']}")
    print(f"HNSW M parameter: {indexing_config['M']}")

    # Initialize pipeline with high accuracy preset
    pipeline = PersonReIDPipeline(config_preset='high_accuracy')
    print("✓ Pipeline initialized with high accuracy preset")


def example_4_save_and_load_configuration():
    """Example 4: Saving and loading configuration"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Saving and Loading Configuration")
    print("=" * 80)

    # Create output directory
    config_dir = Path('output/configs')
    config_dir.mkdir(parents=True, exist_ok=True)

    # Save current configuration
    config_file = config_dir / 'my_config.json'
    config.save_config(str(config_file))
    print(f"✓ Configuration saved to {config_file}")

    # Modify some settings
    update_config('detection', 'confidence_threshold', 0.9)
    update_config('tracking', 'max_disappeared', 50)

    print("Configuration modified:")
    print(
        f"- Detection confidence: {get_config('detection')['confidence_threshold']}")
    print(
        f"- Tracking max disappeared: {get_config('tracking')['max_disappeared']}")

    # Load original configuration
    config.load_config(str(config_file))
    print(f"✓ Configuration loaded from {config_file}")

    print("Configuration restored:")
    print(
        f"- Detection confidence: {get_config('detection')['confidence_threshold']}")
    print(
        f"- Tracking max disappeared: {get_config('tracking')['max_disappeared']}")


def example_5_module_specific_configuration():
    """Example 5: Working with module-specific configurations"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Working with Module-Specific Configurations")
    print("=" * 80)

    # Get all configuration modules
    all_configs = get_config()
    print(f"Available configuration modules: {list(all_configs.keys())}")

    # Focus on detection configuration
    detection_config = get_config('detection')
    print(f"\nDetection configuration has {len(detection_config)} parameters:")

    # Show key detection parameters
    key_params = [
        'model_path', 'confidence_threshold', 'iou_threshold',
        'input_size', 'target_classes'
    ]

    for param in key_params:
        if param in detection_config:
            print(f"- {param}: {detection_config[param]}")

    # Focus on performance configuration
    performance_config = get_config('performance')
    print(f"\nPerformance configuration:")
    print(f"- Optimization level: {performance_config['optimization_level']}")
    print(
        f"- Auto device selection: {performance_config['auto_device_selection']}")
    print(
        f"- GPU memory fraction: {performance_config['gpu_memory_fraction']}")
    print(f"- Enable profiling: {performance_config['enable_profiling']}")


def example_6_runtime_configuration_changes():
    """Example 6: Runtime configuration changes"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Runtime Configuration Changes")
    print("=" * 80)

    # Initialize pipeline
    pipeline = PersonReIDPipeline()

    # Show current HNSW search parameters
    print("Current HNSW search parameters:")
    print(f"- ef_search: {pipeline.index.ef_search}")
    print(f"- default_k: {pipeline.index.default_k}")

    # Update HNSW parameters at runtime
    pipeline.index.set_ef(100)  # Increase search quality
    update_config('indexing', 'ef_search', 100)

    print("\nUpdated HNSW parameters:")
    print(f"- ef_search: {pipeline.index.ef_search}")
    print("✓ Runtime configuration updated")


def example_7_configuration_validation():
    """Example 7: Configuration validation and error handling"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Configuration Validation and Error Handling")
    print("=" * 80)

    try:
        # Try to update a non-existent parameter
        update_config('detection', 'non_existent_param', 0.5)
    except KeyError as e:
        print(f"✓ Caught expected error: {e}")

    try:
        # Try to get configuration for non-existent module
        get_config('non_existent_module')
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    # Validate configuration ranges
    detection_config = get_config('detection')
    confidence = detection_config['confidence_threshold']

    if 0.0 <= confidence <= 1.0:
        print(f"✓ Confidence threshold {confidence} is valid")
    else:
        print(f"❌ Confidence threshold {confidence} is out of range")

    # Show configuration constraints
    print("\nConfiguration constraints:")
    print("- Detection confidence_threshold: 0.0 to 1.0")
    print("- ReID similarity_threshold: 0.0 to 1.0")
    print("- HNSW M parameter: 4 to 48 (recommended: 16)")
    print("- HNSW ef_construction: 100 to 800 (recommended: 200)")


def example_8_performance_optimization():
    """Example 8: Performance optimization through configuration"""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Performance Optimization Through Configuration")
    print("=" * 80)

    # Show different optimization strategies
    print("Performance optimization strategies:")

    print("\n1. Speed optimization:")
    print("   - Use smaller YOLO model (yolov8n.pt)")
    print("   - Reduce input size (416x416)")
    print("   - Use smaller ReID model (osnet_x0_5)")
    print("   - Increase batch sizes")
    print("   - Enable mixed precision")

    print("\n2. Accuracy optimization:")
    print("   - Use larger YOLO model (yolov8l.pt)")
    print("   - Increase input size (1280x1280)")
    print("   - Use larger ReID model (osnet_x1_0)")
    print("   - Increase HNSW parameters (M=32, ef_construction=400)")
    print("   - Enable feature enhancement")

    print("\n3. Memory optimization:")
    print("   - Reduce batch sizes")
    print("   - Limit HNSW max_elements")
    print("   - Enable garbage collection")
    print("   - Use memory mapping for large indices")

    # Apply speed optimization
    print("\nApplying speed optimization preset...")
    apply_preset('high_speed')

    speed_config = get_config()
    print(f"✓ Speed optimization applied")
    print(f"- YOLO model: {speed_config['detection']['model_path']}")
    print(f"- Input size: {speed_config['detection']['input_size']}")
    print(f"- ReID model: {speed_config['reid']['model_name']}")


def main():
    """Run all configuration examples"""
    print("GLOBAL CONFIGURATION SYSTEM EXAMPLES")
    print("=" * 80)
    print("This script demonstrates various ways to use the global configuration system")
    print("for the Person Re-identification System.")

    # Run examples
    example_1_basic_usage()
    example_2_custom_configuration()
    example_3_configuration_presets()
    example_4_save_and_load_configuration()
    example_5_module_specific_configuration()
    example_6_runtime_configuration_changes()
    example_7_configuration_validation()
    example_8_performance_optimization()

    print("\n" + "=" * 80)
    print("CONFIGURATION EXAMPLES COMPLETED")
    print("=" * 80)
    print("For more information, see:")
    print("- config/global_config.py: Main configuration file")
    print("- config/config_documentation.md: Detailed parameter documentation")
    print("- run_demo.py --config-summary: Show current configuration")
    print("- run_demo.py --preset balanced: Use configuration presets")


if __name__ == "__main__":
    main()
