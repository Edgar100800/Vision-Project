#!/usr/bin/env python3
"""
Runner script for finalversion.py
Ensures proper virtual environment usage and provides easy command-line interface
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def check_venv():
    """Check if we're running in the virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return True
    return False


def run_with_venv(script_args):
    """Run the script with virtual environment activated"""
    venv_path = Path("venv/bin/activate")

    if not venv_path.exists():
        print("❌ Virtual environment not found at venv/bin/activate")
        print("Please create a virtual environment first:")
        print("python -m venv venv")
        print("source venv/bin/activate")
        print("pip install -r requirements.txt")
        return False

    # Create command to run with venv
    cmd = f"source venv/bin/activate && python finalversion.py {' '.join(script_args)}"

    print(f"🚀 Running: {cmd}")
    print("=" * 60)

    # Run the command
    result = subprocess.run(cmd, shell=True, cwd=os.getcwd())

    return result.returncode == 0


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run finalversion.py with proper virtual environment setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_finalversion.py data/raw/video.mp4 output/my_test
  python run_finalversion.py data/raw/video.mp4 output/my_test --max_frames 100
  python run_finalversion.py --help-finalversion  # Show finalversion.py help
        """
    )

    parser.add_argument("--help-finalversion", action="store_true",
                        help="Show help for finalversion.py")

    parser.add_argument("--test", action="store_true",
                        help="Run the test suite instead")

    # Parse known args to allow passing through to finalversion.py
    args, unknown = parser.parse_known_args()

    if args.help_finalversion:
        # Show help for finalversion.py
        if check_venv():
            subprocess.run([sys.executable, "finalversion.py", "--help"])
        else:
            subprocess.run(
                "source venv/bin/activate && python finalversion.py --help", shell=True)
        return

    if args.test:
        # Run the test suite
        print("🧪 Running test suite...")
        if check_venv():
            result = subprocess.run([sys.executable, "test_finalversion.py"])
        else:
            result = subprocess.run(
                "source venv/bin/activate && python test_finalversion.py", shell=True)
        return result.returncode == 0

    # Check if we're already in venv
    if check_venv():
        # Already in venv, run directly
        print("✅ Virtual environment already active")
        # Add src to path for imports
        sys.path.insert(0, 'src')
        import finalversion
        sys.argv = ["finalversion.py"] + unknown
        finalversion.main()
    else:
        # Not in venv, run with venv
        print("🔄 Activating virtual environment...")
        success = run_with_venv(unknown)
        if not success:
            print("❌ Script execution failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
