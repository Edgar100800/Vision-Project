#!/usr/bin/env python3
"""
Setup script for the Store Monitoring System.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"🔧 {description}")
    print(f"   Running: {command}")

    try:
        result = subprocess.run(command, shell=True,
                                check=True, capture_output=True, text=True)
        print(f"   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        print(f"   Output: {e.stdout}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False

    print("✅ Python version is compatible")
    return True


def create_directories():
    """Create required directories."""
    directories = [
        "data/raw",
        "data/processed",
        "logs",
        "models",
        "configs"
    ]

    print("📁 Creating directories...")
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")

    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements.txt"):
        return False

    return True


def setup_fastreid():
    """Setup FastReID from source."""
    print("🚀 Setting up FastReID...")

    if os.path.exists("fast-reid"):
        print("   FastReID already exists, skipping clone")
    else:
        if not run_command("git clone https://github.com/JDAI-CV/fast-reid.git", "Cloning FastReID"):
            return False

    # Install FastReID
    if not run_command("cd fast-reid && pip install -r requirements.txt", "Installing FastReID requirements"):
        return False

    if not run_command("cd fast-reid && python setup.py develop", "Installing FastReID"):
        return False

    return True


def download_models():
    """Download pre-trained models."""
    print("🤖 Downloading models...")

    # YOLO model will be downloaded automatically on first use
    print("   YOLO models will be downloaded automatically on first use")

    # Create models directory
    os.makedirs("models", exist_ok=True)
    print("   ✅ Models directory created")


def create_example_config():
    """Create example configuration file."""
    print("⚙️  Creating example configuration...")

    config_content = """# Store Monitoring System Configuration
# Copy this file to config.yaml and modify as needed

detection:
  model_name: "yolov8n.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: "auto"

tracking:
  max_age: 30
  n_init: 3
  max_iou_distance: 0.7
  max_cosine_distance: 0.2

zones:
  entrance_line: [100, 0, 100, 720]
  exit_line: [1180, 0, 1180, 720]
  queue_area: [200, 400, 800, 600]
  cashier_area: [900, 400, 1100, 600]

video:
  input_path: "data/video.mp4"
  output_path: "data/processed/output.mp4"
  fps: 30
"""

    with open("config.example.yaml", "w") as f:
        f.write(config_content)

    print("   ✅ config.example.yaml created")


def main():
    """Main setup function."""
    print("🏪 Store Monitoring System Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create directories
    create_directories()

    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)

    # Setup FastReID
    if not setup_fastreid():
        print("❌ Failed to setup FastReID")
        sys.exit(1)

    # Download models
    download_models()

    # Create example config
    create_example_config()

    print("\n🎉 Setup completed successfully!")
    print("=" * 50)
    print("Next steps:")
    print("1. Add your test video to data/video.mp4")
    print("2. Run: python test_setup.py")
    print("3. If tests pass, run: python src/main.py --video data/video.mp4")
    print("4. For live display, add --display flag")


if __name__ == "__main__":
    main()
